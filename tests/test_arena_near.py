"""
Tester for arena_near: Numba-dekoding av Roaring-blobs og native NEAR,
mot pyroaring og en brute force-referanse.
"""

from __future__ import annotations

import random
import sqlite3

import numpy as np
import pytest
from pyroaring import BitMap

from arena_near import arena_offsets, near_arena, roaring_decode
from bench_rank_unigrams import fetch_and_near_rank_bulk
from convert_shard_to_rank import convert_shard
from test_rank_unigrams import brute_near, doc_bm, make_source_shard


def decode(bm: BitMap) -> list:
    blob = np.frombuffer(bm.serialize(), dtype=np.uint8)
    out = np.empty(max(1, len(bm)), dtype=np.uint32)
    n = roaring_decode(blob, 0, len(blob), out)
    assert n == len(bm)
    return out[:n].tolist()


def bitmaps():
    rnd = random.Random(7)
    yield "empty", BitMap()
    yield "array", BitMap(rnd.sample(range(60_000), 300))
    yield "bitmap", BitMap(rnd.sample(range(65_536), 20_000))
    yield "multi", BitMap(rnd.sample(range(1_000_000), 5_000))
    runs = BitMap(range(100, 5_000))
    runs.update(range(70_000, 71_000))
    runs.run_optimize()
    yield "runs_few", runs
    many = BitMap()
    for c in range(6):  # >= 4 containere med runs gir offset-tabell i formatet
        many.update(range(c * 65_536 + 10, c * 65_536 + 3_000))
        many.update(rnd.sample(range(c * 65_536 + 4_000, c * 65_536 + 60_000), 50))
    many.run_optimize()
    yield "runs_many", many
    mixed = BitMap(rnd.sample(range(65_536), 30_000))
    mixed.update(range(65_536, 65_536 + 8_000))
    mixed.update(rnd.sample(range(200_000, 300_000), 100))
    mixed.run_optimize()
    yield "mixed", mixed


@pytest.mark.parametrize("name,bm", list(bitmaps()))
def test_roaring_decode_matches_pyroaring(name, bm):
    assert decode(bm) == list(bm)


def test_roaring_decode_buffer_too_small():
    bm = BitMap(range(100))
    blob = np.frombuffer(bm.serialize(), dtype=np.uint8)
    assert roaring_decode(blob, 0, len(blob), np.empty(10, dtype=np.uint32)) == -1


def make_arena(per_book: dict) -> tuple:
    docs = BitMap(per_book)
    blobs = [per_book[b].serialize() for b in docs]
    off = np.zeros(len(blobs) + 1, dtype=np.uint32)
    off[1:] = np.cumsum([len(x) for x in blobs])
    return docs, off, b"".join(blobs)


PATHS = {"merge": 0.0, "auto": 4.0, "bitset": 1e9}


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("window", [1, 5, 63])
def test_near_arena_matches_brute_force(seed, window, path):
    rnd = random.Random(seed)
    books = range(50)
    # Ett token per posisjon: to ulike ord kan aldri stå på samme posisjon i samme bok
    A = {b: BitMap(rnd.sample(range(0, 3_000, 2), rnd.randint(1, 80))) for b in rnd.sample(books, 30)}
    B = {b: BitMap(rnd.sample(range(1, 3_000, 2), rnd.randint(1, 80))) for b in rnd.sample(books, 30)}
    da, oa, aa = make_arena(A)
    db, ob, ab = make_arena(B)

    common = sorted(set(A) & set(B))
    hits = [brute_near(A[b], B[b], window) for b in common]
    expected = (sum(hits), sum(1 for h in hits if h))
    assert near_arena(da, oa, aa, db, ob, ab, window=window, dense_factor=PATHS[path]) == expected


@pytest.mark.parametrize("path", list(PATHS))
def test_near_arena_same_word_excludes_self(path):
    A = {3: BitMap([10, 12, 40, 100]), 9: BitMap([5]), 11: BitMap([1, 2, 3])}
    da, oa, aa = make_arena(A)
    # bok 3: 10↔12 er treff, 40 og 100 står alene; bok 9 har én posisjon; bok 11: alle tre
    assert near_arena(da, oa, aa, da, oa, aa, window=5, same_word=True,
                      dense_factor=PATHS[path]) == (5, 2)


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("seed", range(3))
def test_near_arena_dense_containers(seed, path):
    """Tette ord over flere 65 536-blokker: bitmap-, run- og array-containere, og treff over ordgrenser."""
    rnd = random.Random(seed)
    span = 3 * 65_536
    A, B = {}, {}
    for book in range(4):
        pos = list(range(span))
        rnd.shuffle(pos)
        a = BitMap(pos[:40_000])            # tett: bitmap-containere
        b = BitMap(pos[40_000:52_000])      # middels: blanding
        b.update(range(150_000, 150_300))   # en run i tredje blokk
        b -= a
        b.run_optimize()
        A[book], B[book] = a, b
    da, oa, aa = make_arena(A)
    db, ob, ab = make_arena(B)
    expected_hits = [brute_near_fast(A[b], B[b], 5) for b in A]
    expected = (sum(expected_hits), sum(1 for h in expected_hits if h))
    assert near_arena(da, oa, aa, db, ob, ab, window=5, dense_factor=PATHS[path]) == expected
    # samme ord mot seg selv
    exp_self = [brute_near_fast(A[b], A[b], 5, exclude_self=True) for b in A]
    assert near_arena(da, oa, aa, da, oa, aa, window=5, same_word=True,
                      dense_factor=PATHS[path]) == (sum(exp_self), sum(1 for h in exp_self if h))


def test_near_arena_rejects_bad_window():
    da, oa, aa = make_arena({1: BitMap([1])})
    with pytest.raises(ValueError):
        near_arena(da, oa, aa, da, oa, aa, window=64)


def brute_near_fast(a: BitMap, b: BitMap, window: int, exclude_self: bool = False) -> int:
    """Referanse med numpy for store mengder: p i A med en q i B, 0 < |p-q| <= window (eller 0 tillatt)."""
    pa = np.array(list(a), dtype=np.int64)
    pb = np.array(list(b), dtype=np.int64)
    if pb.size == 0:
        return 0
    lo = np.searchsorted(pb, pa - window, side="left")
    hi = np.searchsorted(pb, pa + window, side="right")
    cnt = hi - lo
    if exclude_self:
        cnt = cnt - np.isin(pa, pb).astype(np.int64)
    return int(np.count_nonzero(cnt > 0))


def test_near_arena_disjoint_books():
    da, oa, aa = make_arena({1: BitMap([1])})
    db, ob, ab = make_arena({2: BitMap([1])})
    assert near_arena(da, oa, aa, db, ob, ab) == (0, 0)


def test_near_arena_on_converted_shard(tmp_path):
    """Samme svar som pyroaring-leseren på en shard bygget av convert_shard, også for df == 1."""
    postings = make_source_shard(tmp_path / "src.db", seed=3)
    convert_shard(tmp_path / "src.db", tmp_path / "rank.db", tmp_path / "buckets")
    con = sqlite3.connect(tmp_path / "rank.db")

    def load(cf):
        off_b, arena = con.execute(
            "SELECT offsets, pos_arena FROM word_unigrams WHERE cf_id = ?", (cf,)
        ).fetchone()
        return doc_bm(con, cf), arena_offsets(off_b, len(arena)), arena

    for cf_a, cf_b in [(0, 1), (2, 3), (5, 5), (1000, 1000)]:
        da, oa, aa = load(cf_a)
        db, ob, ab = (da, oa, aa) if cf_a == cf_b else load(cf_b)
        books = sorted(da & db)
        got = near_arena(da, oa, aa, db, ob, ab, window=5, same_word=cf_a == cf_b)
        if cf_a != cf_b:
            ref = fetch_and_near_rank_bulk(con, {"cf_id": cf_a, "doc_bm": da}, {"cf_id": cf_b, "doc_bm": db}, books)
            assert got[0] == ref
        hits = []
        for b in books:
            pa, pb = postings[(b, cf_a)], postings[(b, cf_b)]
            if cf_a == cf_b:
                hits.append(sum(1 for p in pa if any(0 < abs(p - q) <= 5 for q in pa)))
            else:
                hits.append(brute_near(pa, pb, 5))
        assert got == (sum(hits), sum(1 for h in hits if h))
    con.close()
