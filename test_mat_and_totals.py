"""
Test og benchmark av Partielle Funksjoner (Roaring Bitmaps + alignede verdier)
mot Nasjonalbibliotekets totals_500k.parquet og DDK 641 (mat_bundle.json).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyroaring import BitMap

from partial_func import (
    PartialFunc,
    SparseWorkspace,
    _bitblock_intersect_dot_and_jsd,
)

DOC_TO_VEC_DIR = Path("/Users/larsj/Github/doc-to-vec")
TOTALS_PATH = DOC_TO_VEC_DIR / "data" / "totals_500k.parquet"
MAT_BUNDLE_PATH = DOC_TO_VEC_DIR / "mat_bundle.json"


def load_universe_and_books() -> Tuple[
    Dict[str, int],
    List[str],
    PartialFunc,
    PartialFunc,
    List[dict],
    List[PartialFunc],
]:
    """
    1. Nummererer alle ord i totals_500k (1 .. M) sortert etter fallende frekvens.
       Eventuelle ord i mat_bundle som ikke finnes i totals_500k får ID-er på halen.
    2. Bygger bakgrunnsfordelingen b_totals som en PartialFunc.
    3. Bygger hver av de 338 bøkene i mat_bundle.json som en PartialFunc(dom, val).
    """
    t0 = time.perf_counter()
    df = pd.read_parquet(TOTALS_PATH)
    df["word"] = df["word"].astype(str).str.lower()
    df = (
        df.groupby("word", as_index=False)["total_freq"]
        .sum()
        .sort_values("total_freq", ascending=False)
        .reset_index(drop=True)
    )

    # 1-basert nummerering: 1 .. N_totals
    id_to_word: List[str] = ["<PAD>"] + df["word"].tolist()
    word_to_id: Dict[str, int] = {w: i for i, w in enumerate(id_to_word) if i > 0}

    bg_ids = np.arange(1, len(id_to_word), dtype=np.uint32)
    bg_freqs = df["total_freq"].to_numpy(dtype=np.float64)

    with open(MAT_BUNDLE_PATH, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    bundle_words = bundle["words"]
    local_to_global = np.empty(len(bundle_words), dtype=np.uint32)
    extra_ids: List[int] = []
    extra_freqs: List[float] = []

    for local_idx, w_obj in enumerate(bundle_words):
        w_str = w_obj["w"].lower()
        gid = word_to_id.get(w_str)
        if gid is None:
            gid = len(id_to_word)
            id_to_word.append(w_str)
            word_to_id[w_str] = gid
            extra_ids.append(gid)
            extra_freqs.append(float(w_obj.get("tf_bg", 1.0)))
        local_to_global[local_idx] = gid

    if extra_ids:
        bg_ids = np.concatenate([bg_ids, np.array(extra_ids, dtype=np.uint32)])
        bg_freqs = np.concatenate([bg_freqs, np.array(extra_freqs, dtype=np.float64)])

    bg_totals = PartialFunc.from_sorted_arrays(bg_ids, bg_freqs)
    bg_prob = bg_totals.to_prob()

    # Konverter hver bok i CSR-pakken til en PartialFunc over det globale ordrommet
    csr = bundle["csr"]
    row_ptr = csr["row_ptr"]
    col_idx = np.asarray(csr["col_idx"], dtype=np.int64)
    freq = np.asarray(csr["freq"], dtype=np.float64)

    books_meta = bundle["books"]
    book_funcs: List[PartialFunc] = []

    for b_idx in range(len(books_meta)):
        start, end = row_ptr[b_idx], row_ptr[b_idx + 1]
        loc_cols = col_idx[start:end]
        b_freqs = freq[start:end]
        glob_ids = local_to_global[loc_cols]
        # Sorter etter global ord-ID slik at val_f er alignet med dom(f)
        order = np.argsort(glob_ids)
        pf = PartialFunc.from_sorted_arrays(glob_ids[order], b_freqs[order])
        book_funcs.append(pf)

    elapsed = (time.perf_counter() - t0) * 1000
    print(
        f"[Lastet univers & korpus på {elapsed:.1f} ms]\n"
        f"  Globalt ordrom |Words| : {len(id_to_word) - 1:,} ord (ID 1 .. {len(id_to_word) - 1:,})\n"
        f"  Antall bøker (DDK 641) : {len(book_funcs)} bøker\n"
        f"  Snitt |dom(f)| per bok : {np.mean([len(f) for f in book_funcs]):.1f} unike ord\n"
    )
    return word_to_id, id_to_word, bg_totals, bg_prob, books_meta, book_funcs


def verify_vector_space_axioms(book_funcs: List[PartialFunc]) -> None:
    """Verifiserer vektorromsaksiomene på ekte partielle funksjoner uten N-tupler."""
    f, g, h = book_funcs[0], book_funcs[1], book_funcs[2]
    z = PartialFunc.zero()
    a, b = 2.5, -1.25

    def eq(x: PartialFunc, y: PartialFunc, tol: float = 1e-10) -> bool:
        diff = (x - y).prune_zeros(tol=tol)
        return len(diff) == 0

    assert eq(f + g, g + f), "Kommutativitet feilet"
    assert eq((f + g) + h, f + (g + h)), "Assosiativitet feilet"
    assert eq(f + z, f), "Null-element feilet"
    assert eq(f + (-f), z), "Additiv invers feilet"
    assert eq(a * (f + g), a * f + a * g), "Distributivitet over vektoraddisjon feilet"
    assert eq((a + b) * f, a * f + b * f), "Distributivitet over skalaraddisjon feilet"
    assert eq((a * b) * f, a * (b * f)), "Skalar assosiativitet feilet"
    assert eq(1.0 * f, f), "Multiplikativ identitet feilet"

    print("[Vektorromsaksiomer verifisert OK]")
    print(
        f"  |dom(f)| = {len(f)}, |dom(g)| = {len(g)}, "
        f"|dom(f + g)| = {len(f + g)} (union), |dom(f * g)| = {len(f * g)} (snitt)\n"
    )


def inspect_roaring_container_distribution(book_funcs: List[PartialFunc]) -> None:
    """
    Viser hvor mange ord i bøkene som havner i Roaring Container 0 (ID 0..65535,
    de 65 536 vanligste ordene i språket) vs høyere containere (65536+).
    """
    c0_Shares = []
    for f in book_funcs:
        n_total = len(f.dom)
        if n_total > 0:
            n_c0 = f.dom.range_cardinality(0, 65536)
            c0_Shares.append(n_c0 / n_total)
    print("[Frekvensnummerering (1..500k) og Roaring 16-bit containere]")
    print(
        f"  Andel av bokas ord som ligger i Container 0 (ID < 65 536): "
        f"{100 * np.mean(c0_Shares):.1f}% i snitt (min {100 * np.min(c0_Shares):.1f}%, maks {100 * np.max(c0_Shares):.1f}%)\n"
    )


def benchmark_alignment_methods(
    book_funcs: List[PartialFunc],
    universe_size: int,
) -> None:
    """
    Sammenligner hastighet og konsistens mellom de ulike måtene å hente
    verdier på over snittet dom(f) & dom(g).
    """
    # Varm opp Numba-kjerner
    p0 = book_funcs[0].to_prob()
    p1 = book_funcs[1].to_prob()
    _ = p0.align_intersection(p1, method="gallop")
    _ = p0.align_intersection(p1, method="twopointer")
    b0, m0, o0 = p0.bitblocks
    b1, m1, o1 = p1.bitblocks
    _ = _bitblock_intersect_dot_and_jsd(b0, m0, o0, p0.val, b1, m1, o1, p1.val)

    # Forbered sannsynlighetsfordelinger og 64-bits blokker for alle 338 bøker
    probs = [f.to_prob() for f in book_funcs]
    for p in probs:
        _ = p.dom_array
        _ = p.bitblocks

    print("[Benchmark 1: Enkeltbok mot 337 andre bøker (snitt + prikkprodukt + JSD)]")
    anchor = probs[0]

    # 1. Roaring .rank(w) i løkke
    t0 = time.perf_counter()
    res_rank = []
    for g in probs[1:50]:  # 49 par for rank (siden Python-løkke per element er tregere)
        inter_bm, vf, vg = anchor.align_intersection(g, method="rank")
        res_rank.append(float(np.dot(vf, vg)))
    t_rank_per_pair = (time.perf_counter() - t0) / 49 * 1e6

    # 2. Roaring snitt + Galloping / binærsøk
    t0 = time.perf_counter()
    res_gallop = []
    for g in probs[1:]:
        inter_bm, vf, vg = anchor.align_intersection(g, method="gallop")
        res_gallop.append(float(np.dot(vf, vg)))
    t_gallop_per_pair = (time.perf_counter() - t0) / (len(probs) - 1) * 1e6

    # 3. Roaring snitt + To-peker sveip
    t0 = time.perf_counter()
    res_twoptr = []
    for g in probs[1:]:
        inter_bm, vf, vg = anchor.align_intersection(g, method="twopointer")
        res_twoptr.append(float(np.dot(vf, vg)))
    t_twoptr_per_pair = (time.perf_counter() - t0) / (len(probs) - 1) * 1e6

    # 4. 64-bit BitBlock + Hardware POPCNT (0 søk, 0 workspace)
    t0 = time.perf_counter()
    res_popcnt = []
    res_jsd_popcnt = []
    bf, mf, of = anchor.bitblocks
    vf_arr = anchor.val
    for g in probs[1:]:
        bg, mg, og = g.bitblocks
        dot, jsd_sum, tv_sum, _ = _bitblock_intersect_dot_and_jsd(
            bf, mf, of, vf_arr, bg, mg, og, g.val
        )
        res_popcnt.append(dot)
        res_jsd_popcnt.append(1.0 + 0.5 * jsd_sum)
    t_popcnt_per_pair = (time.perf_counter() - t0) / (len(probs) - 1) * 1e6

    # 5. Sparse Accumulator Workspace (Gustavsons 1-til-mange triks)
    ws = SparseWorkspace(universe_size + 10)
    ws.scatter(anchor)
    _ = ws.dot_and_jsd_with_active(probs[1])  # warmup
    t0 = time.perf_counter()
    ws.scatter(anchor)
    res_ws = []
    res_jsd_ws = []
    for g in probs[1:]:
        dot, jsd_v, _, _ = ws.dot_and_jsd_with_active(g)
        res_ws.append(dot)
        res_jsd_ws.append(jsd_v)
    ws.clear()
    t_ws_per_pair = (time.perf_counter() - t0) / (len(probs) - 1) * 1e6

    # Sjekk at alle 5 metoder gir identiske tall!
    assert np.allclose(res_rank, res_gallop[:49])
    assert np.allclose(res_gallop, res_twoptr)
    assert np.allclose(res_gallop, res_popcnt)
    assert np.allclose(res_gallop, res_ws)
    assert np.allclose(res_jsd_popcnt, res_jsd_ws)

    print(f"  1. Roaring bm.rank(w) (Python-iter)  : {t_rank_per_pair:7.2f} µs / bokpar")
    print(f"  2. Roaring & + Galloping search      : {t_gallop_per_pair:7.2f} µs / bokpar")
    print(f"  3. Roaring & + Two-pointer merge     : {t_twoptr_per_pair:7.2f} µs / bokpar")
    print(f"  4. 64-bit BitBlock + POPCNT (dot+JSD): {t_popcnt_per_pair:7.2f} µs / bokpar")
    print(f"  5. Sparse Workspace (dot+JSD)        : {t_ws_per_pair:7.2f} µs / bokpar")

    # Alle-mot-alle (338 * 337 / 2 = 56 953 bokpar) med Sparse Workspace
    t0 = time.perf_counter()
    n_books = len(probs)
    jsd_matrix = np.zeros((n_books, n_books), dtype=np.float64)
    for i in range(n_books):
        pi = probs[i]
        ws.scatter(pi)
        for j in range(i + 1, n_books):
            _, d, _, _ = ws.dot_and_jsd_with_active(probs[j])
            jsd_matrix[i, j] = d
            jsd_matrix[j, i] = d
    ws.clear()
    total_all_pairs_ms = (time.perf_counter() - t0) * 1000
    n_pairs = n_books * (n_books - 1) // 2
    print(
        f"\n[Alle-mot-alle JSD-matrise over {n_pairs:,} bokpar]\n"
        f"  Total tid : {total_all_pairs_ms:.1f} ms ({total_all_pairs_ms * 1000 / n_pairs:.2f} µs per par!)\n"
    )


def showcase_distributional_and_pmi_analysis(
    id_to_word: List[str],
    bg_prob: PartialFunc,
    books_meta: List[dict],
    book_funcs: List[PartialFunc],
) -> None:
    """
    Viser hvordan PMI-transformasjon og distribusjonsavstander (JSD, TV, Hellinger)
    fungerer direkte på partielle funksjoner uten fiksert tuppel-geometri.
    """
    print("[Eksempel: PMI som partiell funksjon og distribusjonsdistanser på DDK 641]")
    target_idx = 0  # Kogebog for hvermand (1890)
    f_target = book_funcs[target_idx]
    p_target = f_target.to_prob()
    meta_t = books_meta[target_idx]

    # Beregn Shifted PPMI (alpha=0.90, k=2.0) som en ny PartialFunc
    ppmi_target = f_target.pmi(bg_prob, alpha=0.90, shift_k=2.0, positive=True)
    top_order = np.argsort(ppmi_target.val)[::-1][:10]
    dom_arr = ppmi_target.dom_array
    top_words = [(id_to_word[int(dom_arr[idx])], ppmi_target.val[idx]) for idx in top_order]

    print(f"  Bok #{target_idx}: «{meta_t['title'][:60]}...» ({meta_t['year']})")
    print(f"  |dom(f)| = {len(f_target)} ord -> |dom(PPMI(f))| = {len(ppmi_target)} spesifikke ord")
    print(
        "  Topp 10 PPMI-ord mot nb.totals: "
        + ", ".join(f"{w} ({v:.2f})" for w, v in top_words)
    )

    # Finn nærmeste naboer i distribusjonsrommet (Jensen-Shannon Divergens)
    dists = []
    for idx, g in enumerate(book_funcs):
        if idx == target_idx:
            continue
        pg = g.to_prob()
        jsd_val = p_target.jsd(pg)
        tv_val = p_target.total_variation(pg)
        hel_val = p_target.hellinger(pg)
        n_shared = f_target.dom.intersection_cardinality(g.dom)
        dists.append((jsd_val, tv_val, hel_val, n_shared, idx))

    dists.sort(key=lambda x: x[0])
    print("\n  Nærmeste bøker målt med ren snitt-basert Jensen-Shannon Divergens (JSD):")
    for rank_i, (jsd_v, tv_v, hel_v, n_sh, b_idx) in enumerate(dists[:5], 1):
        m = books_meta[b_idx]
        print(
            f"    {rank_i}. JSD={jsd_v:.3f} | TV={tv_v:.3f} | Hellinger={hel_v:.3f} "
            f"| Felles ord={n_sh:3d} | ({m['year']}) {m['title'][:55]}"
        )


if __name__ == "__main__":
    word_to_id, id_to_word, bg_totals, bg_prob, books_meta, book_funcs = (
        load_universe_and_books()
    )
    verify_vector_space_axioms(book_funcs)
    inspect_roaring_container_distribution(book_funcs)
    benchmark_alignment_methods(book_funcs, len(id_to_word))
    showcase_distributional_and_pmi_analysis(
        id_to_word, bg_prob, books_meta, book_funcs
    )
