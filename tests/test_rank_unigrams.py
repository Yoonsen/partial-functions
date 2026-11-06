"""
Integrasjonstest for convert_shard_to_rank.py og leserne i bench_rank_unigrams.py
på en liten syntetisk Roaring-shard med samme skjema som imag_roaring_main_*.db.

Kjerneinvarianten: for hvert ord cf_id og hver bok b med ordet er
    pos_arena[offsets[r] : offsets[r + 1]],  r = docpost.rank(b) - 1
nøyaktig posisjons-bloben fra den gamle unigrams-tabellen.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
from pyroaring import BitMap

from bench_rank_unigrams import (
    fetch_and_near_rank_blobopen,
    fetch_and_near_rank_bulk,
    simd_near_count,
)
from convert_shard_to_rank import convert_shard

SRC_SCHEMA = """
CREATE TABLE unigrams (
    book_id INTEGER NOT NULL, cf_id INTEGER NOT NULL, tf INTEGER, post BLOB NOT NULL,
    PRIMARY KEY (book_id, cf_id)
) WITHOUT ROWID;
CREATE TABLE urns (book_id INTEGER NOT NULL PRIMARY KEY) WITHOUT ROWID;
CREATE TABLE urns_postings (id INTEGER NOT NULL PRIMARY KEY, post BLOB NOT NULL) WITHOUT ROWID;
CREATE TABLE words (
    word TEXT NOT NULL PRIMARY KEY, raw_id INTEGER NOT NULL UNIQUE, cf_id INTEGER NOT NULL,
    global_id INTEGER, docfreq INTEGER DEFAULT 0, total_tf INTEGER DEFAULT 0,
    docpost BLOB, docpost_is_complement INTEGER DEFAULT 0
) WITHOUT ROWID;
CREATE TABLE meta (key TEXT NOT NULL PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
"""

Postings = Dict[Tuple[int, int], BitMap]  # (book_id, cf_id) -> posisjoner


def make_source_shard(path: Path, seed: int = 0) -> Postings:
    rnd = random.Random(seed)
    books = list(range(1, 41))
    postings: Postings = {}
    # cf_id 0..39 i mange bøker, cf_id 1000 i nøyaktig én bok (df == 1)
    for cf in range(40):
        for b in rnd.sample(books, rnd.randint(2, 30)):
            postings[(b, cf)] = BitMap(rnd.sample(range(5_000), rnd.randint(1, 60)))
    postings[(17, 1000)] = BitMap([3, 9, 27])

    con = sqlite3.connect(path)
    con.executescript(SRC_SCHEMA)
    rows = [(b, cf, len(bm), bm.serialize()) for (b, cf), bm in postings.items()]
    rnd.shuffle(rows)  # innsettingsrekkefølge skal ikke spille noen rolle
    con.executemany("INSERT INTO unigrams VALUES (?, ?, ?, ?)", rows)
    con.executemany("INSERT INTO urns VALUES (?)", [(b,) for b in books])
    con.execute("INSERT INTO urns_postings VALUES (0, ?)", (BitMap(books).serialize(),))
    for cf in sorted({cf for _, cf in postings}):
        docs = BitMap(b for b, c in postings if c == cf)
        con.execute(
            "INSERT INTO words VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (f"w{cf}", cf, cf, cf, len(docs), 0, docs.serialize()),
        )
    con.execute("INSERT INTO meta VALUES ('layout', 'classic')")
    con.commit()
    con.close()
    return postings


@pytest.fixture(scope="module")
def converted(tmp_path_factory) -> Tuple[sqlite3.Connection, Postings]:
    d = tmp_path_factory.mktemp("shard")
    postings = make_source_shard(d / "src.db")
    convert_shard(d / "src.db", d / "rank.db", d / "buckets")
    con = sqlite3.connect(d / "rank.db")
    yield con, postings
    con.close()


def doc_bm(con: sqlite3.Connection, cf: int) -> BitMap:
    (blob,) = con.execute("SELECT docpost FROM words WHERE cf_id = ?", (cf,)).fetchone()
    return BitMap.deserialize(blob)


def test_one_row_per_cf_id_and_meta(converted):
    con, postings = converted
    n_rows = con.execute("SELECT COUNT(*) FROM word_unigrams").fetchone()[0]
    assert n_rows == len({cf for _, cf in postings})
    assert con.execute("SELECT value FROM meta WHERE key = 'layout'").fetchone()[0] == "rank_unigrams_v1"
    assert con.execute("SELECT COUNT(*) FROM words").fetchone()[0] == n_rows


def test_arena_slices_roundtrip(converted):
    con, postings = converted
    for cf, offsets_blob, arena in con.execute(
        "SELECT cf_id, offsets, pos_arena FROM word_unigrams"
    ):
        docs = doc_bm(con, cf)
        if offsets_blob is None:
            assert len(docs) == 1, "offsets er NULL bare når df == 1"
            (b,) = list(docs)
            assert BitMap.deserialize(arena) == postings[(b, cf)]
            continue
        off = np.frombuffer(offsets_blob, dtype=np.uint32)
        assert len(off) == len(docs) + 1 and off[0] == 0 and off[-1] == len(arena)
        for b in docs:
            r = docs.rank(b) - 1
            assert BitMap.deserialize(arena[off[r]:off[r + 1]]) == postings[(b, cf)]


def test_df1_word_has_null_offsets(converted):
    con, _ = converted
    (off,) = con.execute("SELECT offsets FROM word_unigrams WHERE cf_id = 1000").fetchone()
    assert off is None


# ---------------------------------------------------------------------------
# NEAR-telling og leserne fra benchmarken
# ---------------------------------------------------------------------------

def brute_near(a: BitMap, b: BitMap, window: int) -> int:
    bs = sorted(b)
    return sum(1 for p in a if any(0 < abs(p - q) <= window for q in bs))


@pytest.mark.parametrize("seed", range(5))
def test_simd_near_count_matches_brute_force(seed):
    rnd = random.Random(seed)
    a = BitMap(rnd.sample(range(500), 60))
    b = BitMap(rnd.sample(range(500), 40) + [0, 1])  # nær 0: negative skift skal kuttes
    for window in (1, 3, 5):
        assert simd_near_count(a, b, window) == brute_near(a, b, window)


@pytest.mark.parametrize("reader", [fetch_and_near_rank_blobopen, fetch_and_near_rank_bulk])
def test_readers_match_brute_force(converted, reader):
    con, postings = converted
    w1 = {"cf_id": 0, "doc_bm": doc_bm(con, 0)}
    w2 = {"cf_id": 1, "doc_bm": doc_bm(con, 1)}
    books = sorted(w1["doc_bm"] & w2["doc_bm"])
    assert books, "testdataene skal gi felles bøker"
    expected = sum(brute_near(postings[(b, 0)], postings[(b, 1)], 5) for b in books)
    assert reader(con, w1, w2, books) == expected


@pytest.mark.parametrize("reader", [fetch_and_near_rank_blobopen, fetch_and_near_rank_bulk])
def test_readers_handle_df1_word(converted, reader):
    """offsets = NULL (df == 1) betyr at hele pos_arena er den ene boka."""
    con, postings = converted
    w1 = {"cf_id": 1000, "doc_bm": doc_bm(con, 1000)}
    w2 = {"cf_id": 1000, "doc_bm": doc_bm(con, 1000)}
    (b,) = list(w1["doc_bm"])
    expected = brute_near(postings[(b, 1000)], postings[(b, 1000)], 5)
    assert reader(con, w1, w2, [b]) == expected
