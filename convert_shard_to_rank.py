#!/usr/bin/env python3
"""
Konverterer en Roaring-shard (f.eks. imag_roaring_main_0.db, 6.7 GB) fra:
    unigrams (book_id, cf_id, tf, post) PRIMARY KEY (book_id, cf_id) WITHOUT ROWID
    (77,3 millioner B-tre-rader!)
til «Fortran/Rank-modellen»:
    word_unigrams (
        cf_id     INTEGER PRIMARY KEY,
        offsets   BLOB,            -- NULL når df==1 (hele pos_arena er boka!), ellers uint32[k+1]
        pos_arena BLOB NOT NULL    -- Alle k posisjons-bitmaps tett pakket i book_id-rekkefølge
    )
    (Kun 1 rad per cf_id, indeksert direkte via `r = docpost.rank(book_id) - 1`!)
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

RECORD_HEADER = struct.Struct("<II")  # (cf_id, post_len) — book_id kommer allerede i stigende rekkefølge!
N_BUCKETS = 32


def convert_shard(
    src_path: Path,
    dst_path: Path,
    tmp_dir: Path,
) -> None:
    t_start = time.perf_counter()
    if dst_path.exists():
        dst_path.unlink()

    src = sqlite3.connect(f"file:{src_path}?mode=ro&immutable=1", uri=True)

    # 1. Pass 1: Sekvensiell strømming av `unigrams` ut til 32 bøttefiler
    existing_buckets = [tmp_dir / f"bucket_{b:02d}.bin" for b in range(N_BUCKETS)]
    if all(p.exists() and p.stat().st_size > 0 for p in existing_buckets):
        print(f"[Pass 1/3] Gjenbruker eksisterende {N_BUCKETS} bøttefiler i {tmp_dir.name}!")
    else:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Pass 1/3] Strømmer `unigrams` sekvensielt fra {src_path.name} -> {N_BUCKETS} bøtter ...")
        t0 = time.perf_counter()
        bucket_files = [
            open(tmp_dir / f"bucket_{b:02d}.bin", "wb", buffering=4 * 1024 * 1024)
            for b in range(N_BUCKETS)
        ]

        n_rows = 0
        total_post_bytes = 0
        pack_hdr = RECORD_HEADER.pack
        mask = N_BUCKETS - 1

        cur = src.cursor()
        for cf_id, post in cur.execute("SELECT cf_id, post FROM unigrams"):
            f_out = bucket_files[cf_id & mask]
            p_len = len(post)
            f_out.write(pack_hdr(cf_id, p_len))
            f_out.write(post)
            n_rows += 1
            total_post_bytes += p_len
            if n_rows % 10_000_000 == 0:
                dt = time.perf_counter() - t0
                print(
                    f"  ... {n_rows:,} rader ({total_post_bytes / (1024**3):.2f} GB rå blobs) "
                    f"på {dt:.1f}s ({n_rows / dt:,.0f} rader/s)"
                )

        for f_out in bucket_files:
            f_out.close()

        dt_pass1 = time.perf_counter() - t0
        print(
            f"  Pass 1 ferdig på {dt_pass1:.1f}s: {n_rows:,} rader, "
            f"{total_post_bytes / (1024**2):,.1f} MB rå posisjons-blobs."
        )

    # 2. Opprett mål-databasen og kopier metadata/ordbok via read-only URI
    print(f"\n[Pass 2/3] Oppretter {dst_path.name} og kopierer urns + words ...")
    t0 = time.perf_counter()
    dst = sqlite3.connect(str(dst_path), uri=True)
    dst.execute("PRAGMA page_size = 4096")
    dst.execute("PRAGMA journal_mode = OFF")
    dst.execute("PRAGMA synchronous = OFF")
    dst.execute("PRAGMA temp_store = MEMORY")
    dst.execute("PRAGMA cache_size = -500000")

    dst.executescript(
        """
        CREATE TABLE word_unigrams (
            cf_id     INTEGER PRIMARY KEY,
            offsets   BLOB,
            pos_arena BLOB NOT NULL
        );

        CREATE TABLE words (
            word TEXT NOT NULL PRIMARY KEY,
            raw_id INTEGER NOT NULL UNIQUE,
            cf_id INTEGER NOT NULL,
            global_id INTEGER,
            docfreq INTEGER DEFAULT 0,
            total_tf INTEGER DEFAULT 0,
            docpost BLOB,
            docpost_is_complement INTEGER DEFAULT 0
        ) WITHOUT ROWID;

        CREATE TABLE urns (
            book_id INTEGER NOT NULL PRIMARY KEY
        ) WITHOUT ROWID;

        CREATE TABLE urns_postings (
            id INTEGER NOT NULL PRIMARY KEY,
            post BLOB NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE meta (
            key TEXT NOT NULL PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )

    dst.execute("ATTACH DATABASE ? AS srcdb", (f"file:{src_path}?mode=ro&immutable=1",))
    dst.execute("INSERT INTO urns(book_id) SELECT book_id FROM srcdb.urns")
    dst.execute("INSERT INTO urns_postings(id, post) SELECT id, post FROM srcdb.urns_postings")
    dst.execute(
        """
        INSERT INTO words(word, raw_id, cf_id, global_id, docfreq, total_tf, docpost, docpost_is_complement)
        SELECT word, raw_id, cf_id, global_id, docfreq, total_tf, docpost, docpost_is_complement
        FROM srcdb.words
        """
    )
    dst.execute("CREATE INDEX words_cf_id ON words(cf_id)")
    dst.execute("INSERT OR REPLACE INTO meta(key, value) SELECT key, value FROM srcdb.meta")
    dst.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('layout', 'rank_unigrams_v1')")
    dst.commit()
    dst.execute("DETACH DATABASE srcdb")
    src.close()
    print(f"  Kopierte urns og words på {time.perf_counter() - t0:.1f}s.")

    # 3. Pass 3: Bygg `word_unigrams` bøtte for bøtte
    print(f"\n[Pass 3/3] Pakker `word_unigrams` (1 rad per cf_id) fra de {N_BUCKETS} bøttene ...")
    t0 = time.perf_counter()
    unpack_hdr = RECORD_HEADER.unpack_from
    hdr_size = RECORD_HEADER.size
    total_cf_ids = 0
    single_doc_cf_ids = 0

    for b_idx in range(N_BUCKETS):
        b_path = tmp_dir / f"bucket_{b_idx:02d}.bin"
        raw_data = b_path.read_bytes()
        b_path.unlink()  # Frigjør diskplass fortløpende!

        mv = memoryview(raw_data)
        pos = 0
        limit = len(raw_data)

        lens_map: Dict[int, List[int]] = defaultdict(list)
        blobs_map: Dict[int, List[bytes]] = defaultdict(list)

        while pos < limit:
            cf_id, p_len = unpack_hdr(mv, pos)
            pos += hdr_size
            p_bytes = bytes(mv[pos : pos + p_len])
            pos += p_len

            lens_map[cf_id].append(p_len)
            blobs_map[cf_id].append(p_bytes)

        del mv, raw_data

        batch_rows = []
        for cf_id in sorted(blobs_map.keys()):
            b_list = blobs_map[cf_id]
            if len(b_list) == 1:
                # Når ordet kun finnes i 1 bok (df=1), trenger vi ikke offsets-array!
                single_doc_cf_ids += 1
                batch_rows.append((cf_id, None, b_list[0]))
            else:
                l_list = lens_map[cf_id]
                lengths_arr = np.array(l_list, dtype=np.uint32)
                offsets_arr = np.empty(len(lengths_arr) + 1, dtype=np.uint32)
                offsets_arr[0] = 0
                np.cumsum(lengths_arr, dtype=np.uint32, out=offsets_arr[1:])
                batch_rows.append((cf_id, offsets_arr.tobytes(), b"".join(b_list)))

        dst.executemany(
            "INSERT INTO word_unigrams (cf_id, offsets, pos_arena) VALUES (?, ?, ?)",
            batch_rows,
        )
        dst.commit()
        total_cf_ids += len(batch_rows)
        if (b_idx + 1) % 4 == 0 or b_idx == N_BUCKETS - 1:
            print(
                f"  Bøtte {b_idx + 1:2d}/{N_BUCKETS} ferdig -> akkumulert: {total_cf_ids:,} cf_id "
                f"({single_doc_cf_ids:,} med df=1) på {time.perf_counter() - t0:.1f}s"
            )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    dst.execute("ANALYZE")
    dst.commit()
    dst.close()

    src_gb = src_path.stat().st_size / (1024**3)
    dst_gb = dst_path.stat().st_size / (1024**3)
    total_s = time.perf_counter() - t_start

    print("=" * 76)
    print(f"KONVERTERING FERDIG på {total_s:.1f} sekunder!")
    print(f"  Kilde-shard ({src_path.name}) : {src_gb:.2f} GB (77 303 348 rader i unigrams)")
    print(f"  Rank-shard  ({dst_path.name}) : {dst_gb:.2f} GB ({total_cf_ids:,} rader i word_unigrams)")
    print(f"  Spart plass på disk          : {src_gb - dst_gb:.2f} GB ({100 * (1 - dst_gb / src_gb):.1f}% reduksjon!)")
    print("=" * 76)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        default="/Users/larsj/Github/metadata-roaring/shards/imag_roaring_main_0.db",
    )
    p.add_argument(
        "--dst",
        default="/Users/larsj/Github/partial-functions/imag_roaring_rank_0.db",
    )
    p.add_argument(
        "--tmp-dir",
        default="/Users/larsj/Github/partial-functions/_tmp_buckets",
    )
    args = p.parse_args()
    convert_shard(
        Path(args.src),
        Path(args.dst),
        Path(args.tmp_dir),
    )
