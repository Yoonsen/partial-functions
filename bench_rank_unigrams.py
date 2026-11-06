"""
Benchmark: Klassisk SQLite B-tre `PRIMARY KEY (book_id, cf_id)` i `unigrams`
           MOT
           «Fortran/Rank-modellen» `PRIMARY KEY (cf_id)` + `doc_bm.rank(book_id)`!

Tester på ekte data fra Shard 0 (`imag_roaring_main_0.db`, 8 906 bøker fra 1800-tallet):
  1. Lagringsstørrelse og antall B-tre-rader.
  2. Oppslags- og NEAR_5-hastighet både ved:
     - Full Doc-snitt (alle bøker som har begge ord)
     - Chunk-filtrert snitt (kun overlevende bøker fra chunk-sidecar, L=1024, ovl=64)
  3. Sammenligner:
     - Dagens modell: `SELECT post FROM unigrams WHERE book_id = ? AND cf_id = ?`
     - Rank + SQLite `blobopen`: 1 B-tre-oppslag per `cf_id`, deretter direkte
       byte-slice `blob[offsets[r] : offsets[r+1]]` via `r = doc_bm.rank(book_id) - 1`.
     - Rank + Adaptiv (bruker `blobopen` når vi bare trenger noen få bøker, og
       én samlet `SELECT pos_arena` når mange bøker overlever).
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pyroaring import BitMap

SHARD_DB = Path("/Users/larsj/Github/metadata-roaring/shards/imag_roaring_main_0.db")
TMP_RANK_DB = Path("/tmp/bench_rank_unigrams.sqlite")

K_SHIFT = 11
CHUNK_SHIFT = 10
CHUNK_SIZE = 1 << CHUNK_SHIFT
OVERLAP = 64
STRIDE = CHUNK_SIZE - OVERLAP

TEST_WORDS = [
    "penger",
    "kjerlighed",
    "demokrati",
    "frihed",
    "kongen",
    "norge",
    "gud",
    "lov",
    "ret",
]

TEST_PAIRS = [
    ("penger", "kjerlighed"),
    ("demokrati", "frihed"),
    ("penger", "kongen"),
    ("frihed", "kongen"),
    ("norge", "kongen"),
    ("gud", "kongen"),
    ("lov", "ret"),
]


def simd_near_count(bm1: BitMap, bm2: BitMap, window: int = 5) -> int:
    shifted = [bm2.shift(d) for d in range(-window, window + 1) if d != 0]
    neighborhood = BitMap.union(*shifted)
    return bm1.intersection_cardinality(neighborhood)


def build_test_databases(con_src: sqlite3.Connection) -> Tuple[Dict[str, dict], sqlite3.Connection]:
    """
    Henter de 9 testordene fra Shard 0, bygger chunk-bitmaps (L=1024, ovl=64),
    og oppretter en ny SQLite-database med Rank-indeksert `word_unigrams`:
        CREATE TABLE word_unigrams (
            cf_id INTEGER PRIMARY KEY,
            offsets BLOB NOT NULL,
            pos_arena BLOB NOT NULL
        )
    Merk: I SQLite er `cf_id INTEGER PRIMARY KEY` selve rowid-en i B-treet!
    """
    if TMP_RANK_DB.exists():
        TMP_RANK_DB.unlink()

    con_rank = sqlite3.connect(str(TMP_RANK_DB))
    con_rank.execute("PRAGMA page_size = 4096")
    con_rank.execute("PRAGMA journal_mode = OFF")
    con_rank.execute("PRAGMA synchronous = OFF")
    con_rank.execute(
        """
        CREATE TABLE word_unigrams (
            cf_id INTEGER PRIMARY KEY,
            offsets BLOB NOT NULL,
            pos_arena BLOB NOT NULL
        )
        """
    )

    all_books_blob = con_src.execute("SELECT post FROM urns_postings WHERE id = 1").fetchone()[0]
    all_books_bm = BitMap.deserialize(all_books_blob)
    book_list = list(all_books_bm)
    book_to_doc = {b: i for i, b in enumerate(book_list)}

    vocab: Dict[str, dict] = {}
    total_btree_rows = 0
    total_raw_post_bytes = 0
    total_offsets_bytes = 0

    t0 = time.perf_counter()
    for w in TEST_WORDS:
        row = con_src.execute(
            "SELECT cf_id, docfreq, total_tf, docpost, docpost_is_complement FROM words WHERE word = ?",
            (w,),
        ).fetchone()
        cf_id, df, tf, blob, is_comp = row
        cf_id = int(cf_id)
        bm_books = BitMap.deserialize(blob)
        if is_comp:
            bm_books = all_books_bm - bm_books

        # Bygg rank-alignet offsets + pos_arena i NØYAKTIG samme rekkefølge som bm_books!
        # Vi filtrerer bm_books til kun de book_id som faktisk har en rad i unigrams
        actual_books = BitMap()
        blobs_list: List[bytes] = []
        chunk_bm = BitMap()

        for b in bm_books:
            r = con_src.execute(
                "SELECT post FROM unigrams WHERE book_id = ? AND cf_id = ?",
                (b, cf_id),
            ).fetchone()
            if not r or not r[0]:
                continue
            p_blob = bytes(r[0])
            actual_books.add(b)
            blobs_list.append(p_blob)

            # Bygg samtidig chunk_bm (L=1024, ovl=64) for top-down chunk-filtrering
            d = book_to_doc[b]
            base = d << K_SHIFT
            pos_bm = BitMap.deserialize(p_blob)
            for p in pos_bm:
                c0 = p // STRIDE
                if p - c0 * STRIDE < CHUNK_SIZE:
                    chunk_bm.add(base | c0)
                if c0 > 0 and (p - (c0 - 1) * STRIDE) < CHUNK_SIZE:
                    chunk_bm.add(base | (c0 - 1))

        # Bygg offsets-array: uint32[len(actual_books) + 1]
        lengths = np.array([len(b_bytes) for b_bytes in blobs_list], dtype=np.uint32)
        offsets = np.empty(len(lengths) + 1, dtype=np.uint32)
        offsets[0] = 0
        np.cumsum(lengths, dtype=np.uint32, out=offsets[1:])

        pos_arena = b"".join(blobs_list)
        offsets_bytes = offsets.tobytes()

        con_rank.execute(
            "INSERT INTO word_unigrams (cf_id, offsets, pos_arena) VALUES (?, ?, ?)",
            (cf_id, offsets_bytes, pos_arena),
        )

        total_btree_rows += len(actual_books)
        total_raw_post_bytes += len(pos_arena)
        total_offsets_bytes += len(offsets_bytes)

        vocab[w] = {
            "word": w,
            "cf_id": cf_id,
            "doc_bm": actual_books,  # BitMap over book_id (dhlabid)
            "chunk_bm": chunk_bm,    # BitMap over (doc_idx << 11) | local_chunk
            "doc_to_book": book_list,
            "arena_bytes": len(pos_arena),
        }

    con_rank.commit()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Sjekk verifikasjon av rank-indeksering for første ord
    w0 = vocab["demokrati"]
    cf0 = w0["cf_id"]
    off_blob = con_rank.execute("SELECT offsets FROM word_unigrams WHERE cf_id = ?", (cf0,)).fetchone()[0]
    offs = np.frombuffer(off_blob, dtype=np.uint32)
    blob_handle = con_rank.blobopen("word_unigrams", "pos_arena", cf0, readonly=True)
    for test_b in list(w0["doc_bm"])[:10]:
        r_idx = w0["doc_bm"].rank(test_b) - 1
        slice_bytes = blob_handle[int(offs[r_idx]) : int(offs[r_idx + 1])]
        orig_bytes = con_src.execute(
            "SELECT post FROM unigrams WHERE book_id = ? AND cf_id = ?", (test_b, cf0)
        ).fetchone()[0]
        assert slice_bytes == orig_bytes, "Rank-slice samsvarer ikke med B-tre!"
    blob_handle.close()

    print(f"[Bygget og verifisert Rank-indeksert `word_unigrams` på {elapsed_ms:.1f} ms]")
    print(f"  Klassisk B-tre (`unigrams`) : {total_btree_rows:,} separate B-tre-rader for disse 9 ordene")
    print(f"  Rank-modell (`word_unigrams`): KUN 9 rader i B-treet (1 rad per cf_id!)")
    print(f"  Rå posisjons-blobs          : {total_raw_post_bytes / 1024:.1f} KB")
    print(f"  Rank offsets-tabell         : {total_offsets_bytes / 1024:.1f} KB (+{100 * total_offsets_bytes / total_raw_post_bytes:.2f}% for O(1) rank-oppslag)\n")

    return vocab, con_rank


def fetch_and_near_classic_btree(
    con_src: sqlite3.Connection,
    w1: dict,
    w2: dict,
    target_books: List[int],
) -> int:
    """Dagens metode: 1 B-tre oppslag i `unigrams (book_id, cf_id)` per bok per ord."""
    cf1, cf2 = w1["cf_id"], w2["cf_id"]
    hits = 0
    cur = con_src.cursor()
    for b in target_books:
        r1 = cur.execute(
            "SELECT post FROM unigrams WHERE book_id = ? AND cf_id = ?", (b, cf1)
        ).fetchone()
        r2 = cur.execute(
            "SELECT post FROM unigrams WHERE book_id = ? AND cf_id = ?", (b, cf2)
        ).fetchone()
        if r1 and r2:
            bm1 = BitMap.deserialize(r1[0])
            bm2 = BitMap.deserialize(r2[0])
            hits += simd_near_count(bm1, bm2, window=5)
    return hits


def fetch_and_near_rank_blobopen(
    con_rank: sqlite3.Connection,
    w1: dict,
    w2: dict,
    target_books: List[int],
) -> int:
    """
    Rank-indeksert metode med SQLite inkrementell BLOB I/O (`blobopen`):
    - Kun 1 B-tre oppslag per ord (for å hente `offsets` og åpne `pos_arena`-handle).
    - For hver overlevende bok b:
        r1 = doc_bm1.rank(b) - 1
        r2 = doc_bm2.rank(b) - 1
      og direkte byte-slice `blob1[off1[r1] : off1[r1+1]]`!
    """
    if not target_books:
        return 0
    cf1, cf2 = w1["cf_id"], w2["cf_id"]
    dbm1, dbm2 = w1["doc_bm"], w2["doc_bm"]

    off1_blob = con_rank.execute("SELECT offsets FROM word_unigrams WHERE cf_id = ?", (cf1,)).fetchone()[0]
    off2_blob = con_rank.execute("SELECT offsets FROM word_unigrams WHERE cf_id = ?", (cf2,)).fetchone()[0]
    off1 = np.frombuffer(off1_blob, dtype=np.uint32)
    off2 = np.frombuffer(off2_blob, dtype=np.uint32)

    b1 = con_rank.blobopen("word_unigrams", "pos_arena", cf1, readonly=True)
    b2 = con_rank.blobopen("word_unigrams", "pos_arena", cf2, readonly=True)

    hits = 0
    try:
        for b in target_books:
            r1 = dbm1.rank(b) - 1
            r2 = dbm2.rank(b) - 1
            bm1 = BitMap.deserialize(b1[int(off1[r1]) : int(off1[r1 + 1])])
            bm2 = BitMap.deserialize(b2[int(off2[r2]) : int(off2[r2 + 1])])
            hits += simd_near_count(bm1, bm2, window=5)
    finally:
        b1.close()
        b2.close()
    return hits


def fetch_and_near_rank_bulk(
    con_rank: sqlite3.Connection,
    w1: dict,
    w2: dict,
    target_books: List[int],
) -> int:
    """
    Rank-indeksert metode der vi henter (offsets, pos_arena) i ett SQL-kall per ord
    (eller bruker memoryview) og indekserer direkte i bufferen via rank.
    """
    if not target_books:
        return 0
    cf1, cf2 = w1["cf_id"], w2["cf_id"]
    dbm1, dbm2 = w1["doc_bm"], w2["doc_bm"]

    off1_b, arena1 = con_rank.execute(
        "SELECT offsets, pos_arena FROM word_unigrams WHERE cf_id = ?", (cf1,)
    ).fetchone()
    off2_b, arena2 = con_rank.execute(
        "SELECT offsets, pos_arena FROM word_unigrams WHERE cf_id = ?", (cf2,)
    ).fetchone()

    off1 = np.frombuffer(off1_b, dtype=np.uint32)
    off2 = np.frombuffer(off2_b, dtype=np.uint32)
    mv1 = memoryview(arena1)
    mv2 = memoryview(arena2)

    hits = 0
    for b in target_books:
        r1 = dbm1.rank(b) - 1
        r2 = dbm2.rank(b) - 1
        bm1 = BitMap.deserialize(mv1[int(off1[r1]) : int(off1[r1 + 1])])
        bm2 = BitMap.deserialize(mv2[int(off2[r2]) : int(off2[r2 + 1])])
        hits += simd_near_count(bm1, bm2, window=5)
    return hits


def main() -> None:
    con_src = sqlite3.connect(f"file:{SHARD_DB}?mode=ro&immutable=1", uri=True)
    vocab, con_rank = build_test_databases(con_src)

    print("=" * 92)
    print("BENCHMARK (Med Chunk-Sidecar filtrering): Klassisk (book_id, cf_id) B-tre vs. Rank-modell")
    print("=" * 92)
    print(
        f"{'Ordpar':22s} | {'SurvDocs':>8s} | {'B-tre (book,cf)':>15s} | "
        f"{'Rank+blobopen':>14s} | {'Rank+bulk':>11s} | {'Speedup':>8s} | {'NEAR_5':>6s}"
    )
    print("-" * 92)

    for w1_str, w2_str in TEST_PAIRS:
        w1, w2 = vocab[w1_str], vocab[w2_str]
        # Finn overlevende bøker fra Chunk-snittet (slik produksjonssøket gjør)
        shared_chunks = w1["chunk_bm"] & w2["chunk_bm"]
        doc_to_book = w1["doc_to_book"]
        surv_doc_indices = sorted({c >> K_SHIFT for c in shared_chunks})
        surv_books = [doc_to_book[d] for d in surv_doc_indices]

        # Varm opp og kjør klassisk B-tre
        _ = fetch_and_near_classic_btree(con_src, w1, w2, surv_books[:2])
        t0 = time.perf_counter()
        hits_btree = fetch_and_near_classic_btree(con_src, w1, w2, surv_books)
        ms_btree = (time.perf_counter() - t0) * 1000

        # Kjør Rank + SQLite blobopen (kun 1 B-tre oppslag per ord, punktlesing av slices)
        t0 = time.perf_counter()
        hits_blobopen = fetch_and_near_rank_blobopen(con_rank, w1, w2, surv_books)
        ms_blobopen = (time.perf_counter() - t0) * 1000

        # Kjør Rank + Bulk memoryview
        t0 = time.perf_counter()
        hits_bulk = fetch_and_near_rank_bulk(con_rank, w1, w2, surv_books)
        ms_bulk = (time.perf_counter() - t0) * 1000

        assert hits_btree == hits_blobopen == hits_bulk, "Avvik i NEAR_5 treff!"
        best_rank_ms = min(ms_blobopen, ms_bulk)
        speedup = ms_btree / best_rank_ms if best_rank_ms > 0 else 1.0

        pair_label = f"{w1_str} + {w2_str}"
        print(
            f"{pair_label:22s} | {len(surv_books):8d} | {ms_btree:12.2f} ms | "
            f"{ms_blobopen:11.2f} ms | {ms_bulk:8.2f} ms | {speedup:7.2f}x | {hits_btree:6d}"
        )

    con_src.close()
    con_rank.close()


if __name__ == "__main__":
    main()
