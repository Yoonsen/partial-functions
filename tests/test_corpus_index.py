"""
Tester for RoaringCorpusIndex og AlgebraCodec på et lite syntetisk korpus,
der hver størrelse også regnes ut med brute force over (bok, chunk, ord).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest
from pyroaring import BitMap

from codec_and_traceability import AlgebraCodec
from collocations_and_trends import K_SHIFT, RoaringCorpusIndex
from conftest import as_dict, assert_pf_close
from partial_func import PartialFunc

WORDS = [f"ord{i}" for i in range(30)]
YEARS = list(range(1900, 1920))


@dataclass
class Synthetic:
    idx: RoaringCorpusIndex
    chunk_words: Dict[int, Set[int]]   # chunk_id -> ord-ID-er i chunken
    chunk_year: Dict[int, int]
    doc_year: Dict[int, int]
    books_meta: List[dict]


def build_synthetic(seed: int = 0, n_docs: int = 60) -> Synthetic:
    rng = np.random.default_rng(seed)
    word_to_id = {w: i for i, w in enumerate(WORDS)}

    chunk_words: Dict[int, Set[int]] = {}
    chunk_year: Dict[int, int] = {}
    doc_year: Dict[int, int] = {}
    for d in range(n_docs):
        year = YEARS[d % len(YEARS)]
        doc_year[d] = year
        for c in range(int(rng.integers(1, 5))):
            cid = (d << K_SHIFT) | c
            # Zipf-aktig: lave ord-ID-er er vanlige
            k = int(rng.integers(2, 10))
            ws = rng.choice(len(WORDS), size=k, replace=False, p=_zipf(len(WORDS)))
            chunk_words[cid] = {int(w) for w in ws}
            chunk_year[cid] = year

    word_chunk = defaultdict(BitMap)
    word_doc = defaultdict(BitMap)
    for cid, ws in chunk_words.items():
        for w in ws:
            word_chunk[w].add(cid)
            word_doc[w].add(cid >> K_SHIFT)

    year_doc, year_chunk = defaultdict(BitMap), defaultdict(BitMap)
    dec_doc, dec_chunk = defaultdict(BitMap), defaultdict(BitMap)
    for d, y in doc_year.items():
        year_doc[y].add(d)
        dec_doc[y // 10 * 10].add(d)
    for cid, y in chunk_year.items():
        year_chunk[y].add(cid)
        dec_chunk[y // 10 * 10].add(cid)

    def tau(buckets) -> PartialFunc:
        return PartialFunc.from_dict({t: float(len(b)) for t, b in buckets.items()})

    idx = RoaringCorpusIndex(
        word_to_id=word_to_id,
        id_to_word=list(WORDS),
        word_doc_bm=dict(word_doc),
        word_chunk_bm=dict(word_chunk),
        all_docs_bm=BitMap(doc_year),
        all_chunks_bm=BitMap(chunk_words),
        year_doc_bm=dict(year_doc),
        year_chunk_bm=dict(year_chunk),
        decade_doc_bm=dict(dec_doc),
        decade_chunk_bm=dict(dec_chunk),
        tau_year_chunks=tau(year_chunk),
        tau_decade_chunks=tau(dec_chunk),
        corpus_chunk_bg=PartialFunc.from_dict({w: float(len(b)) for w, b in word_chunk.items()}),
    )
    books_meta = [{"title": f"Bok {d}", "year": doc_year[d]} for d in range(n_docs)]
    return Synthetic(idx, chunk_words, chunk_year, doc_year, books_meta)


def _zipf(n: int) -> np.ndarray:
    p = 1.0 / np.arange(1, n + 1)
    return p / p.sum()


@pytest.fixture(params=[0, 1, 2])
def syn(request) -> Synthetic:
    return build_synthetic(seed=request.param)


def brute_profile(syn: Synthetic, A: BitMap) -> Dict[int, float]:
    counts: Dict[int, float] = defaultdict(float)
    for cid in A:
        for w in syn.chunk_words.get(cid, ()):
            counts[w] += 1.0
    return dict(counts)


def brute_trend(syn: Synthetic, A: BitMap, by_decade: bool) -> Dict[int, float]:
    counts: Dict[int, float] = defaultdict(float)
    for cid in A:
        y = syn.chunk_year[cid]
        counts[y // 10 * 10 if by_decade else y] += 1.0
    return dict(counts)


# ---------------------------------------------------------------------------
# Oppslag
# ---------------------------------------------------------------------------

def test_word_lookup_case_insensitive_and_unknown(syn):
    idx = syn.idx
    assert idx.word_chunks("ORD0") == idx.word_chunk_bm[0]
    assert idx.word_docs("ord0") == idx.word_doc_bm[0]
    assert len(idx.word_chunks("finnes_ikke")) == 0
    assert len(idx.word_docs("finnes_ikke")) == 0


# ---------------------------------------------------------------------------
# Kollokasjoner
# ---------------------------------------------------------------------------

def test_collocation_profile_matches_brute_force(syn):
    A = syn.idx.word_chunks("ord3")
    assert_pf_close(syn.idx.collocation_profile(A), brute_profile(syn, A))


def test_collocation_profile_filters(syn):
    A = syn.idx.word_chunks("ord1")
    ref = brute_profile(syn, A)
    got = syn.idx.collocation_profile(A, min_cooc=3, exclude_ids=[1], candidate_ids=range(0, 20))
    assert_pf_close(got, {w: c for w, c in ref.items() if c >= 3 and w != 1 and w < 20})
    assert len(syn.idx.collocation_profile(BitMap())) == 0


@pytest.mark.parametrize("use_mask", [False, True])
def test_collocation_pmi(syn, use_mask):
    idx = syn.idx
    K = idx.docs_to_chunks(BitMap(range(0, 60, 2))) if use_mask else idx.all_chunks_bm
    A = idx.word_chunks("ord2") & K
    alpha = 0.75

    raw, pmi = idx.collocation_pmi(
        idx.word_chunks("ord2"), corpus_mask=K if use_mask else None, min_cooc=2, alpha=alpha
    )
    prof_A = {w: c for w, c in brute_profile(syn, A).items() if c >= 2}
    prof_K = brute_profile(syn, K)
    assert_pf_close(raw, prof_A)
    ref = {
        w: math.log2((c / len(A)) / (prof_K[w] / len(K)) ** alpha)
        for w, c in prof_A.items()
    }
    assert_pf_close(pmi, ref, rtol=1e-9)


# ---------------------------------------------------------------------------
# Trendlinjer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("by_decade", [True, False])
def test_trendline_matches_brute_force(syn, by_decade):
    A = syn.idx.word_chunks("ord5")
    assert_pf_close(syn.idx.trendline(A, by_decade=by_decade), brute_trend(syn, A, by_decade))


@pytest.mark.parametrize("by_decade", [True, False])
def test_relative_trendline(syn, by_decade):
    A = syn.idx.word_chunks("ord0")
    num = brute_trend(syn, A, by_decade)
    tot = brute_trend(syn, syn.idx.all_chunks_bm, by_decade)
    assert_pf_close(syn.idx.relative_trendline(A, by_decade=by_decade),
                    {t: num[t] / tot[t] for t in num})


def test_diachronic_pmi_trendline(syn):
    idx = syn.idx
    x, y = idx.word_chunks("ord0"), idx.word_chunks("ord1")
    tau_xy, pmi = idx.diachronic_pmi_trendline(x, y, by_decade=False)
    t_xy = brute_trend(syn, x & y, False)
    t_x, t_y = brute_trend(syn, x, False), brute_trend(syn, y, False)
    t_tot = brute_trend(syn, idx.all_chunks_bm, False)
    assert_pf_close(tau_xy, t_xy)
    assert_pf_close(pmi, {t: math.log2(t_xy[t] / (t_x[t] * t_y[t] / t_tot[t])) for t in t_xy},
                    rtol=1e-9)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sample_docs_reproducible_and_subset(syn):
    idx = syn.idx
    s = idx.sample_docs(20, seed=42)
    assert len(s) == 20 and s.issubset(idx.all_docs_bm)
    assert s == idx.sample_docs(20, seed=42)
    assert s != idx.sample_docs(20, seed=43)


@pytest.mark.parametrize("by", ["year", "decade"])
def test_sample_docs_stratified(syn, by):
    idx = syn.idx
    buckets = idx.year_doc_bm if by == "year" else idx.decade_doc_bm
    s = idx.sample_docs(2, by=by, seed=1)
    for b in buckets.values():
        assert len(s & b) == min(2, len(b))


def test_sample_docs_respects_corpus_and_small_strata(syn):
    idx = syn.idx
    C = BitMap(range(0, 60, 7))
    s = idx.sample_docs(100, corpus=C, by="year", seed=0)
    assert s == C, "når n er større enn stratumet, tas hele stratumet med"
    assert len(idx.sample_docs(5, corpus=BitMap())) == 0


def test_sample_docs_invalid_by(syn):
    with pytest.raises(ValueError):
        syn.idx.sample_docs(5, by="month")


def test_docs_to_chunks(syn):
    docs = BitMap([0, 7, 59])
    expected = BitMap(c for c in syn.chunk_words if (c >> K_SHIFT) in docs)
    assert syn.idx.docs_to_chunks(docs) == expected
    assert len(syn.idx.docs_to_chunks(BitMap())) == 0


# ---------------------------------------------------------------------------
# AlgebraCodec: inn-koding og dekoding
# ---------------------------------------------------------------------------

@pytest.fixture
def codec(syn) -> Tuple[AlgebraCodec, Synthetic]:
    c = AlgebraCodec(syn.idx, syn.books_meta, book_funcs=[], bg_prob=PartialFunc.zero())
    return c, syn


def test_encode_word_id(codec):
    c, _ = codec
    assert c.encode_word_id("Ord4") == 4
    with pytest.raises(KeyError):
        c.encode_word_id("finnes_ikke")
    assert c.encode_words_bitmap(["ord1", "ORD2", "finnes_ikke"]) == BitMap([1, 2])


def test_encode_chunk_context_matches_brute_force(codec):
    c, syn = codec
    got = c.encode_chunk_context(
        all_of=["ord0"], any_of=["ord1", "ord2"], none_of=["ord3"], year_range=(1905, 1914)
    )
    expected = BitMap(
        cid for cid, ws in syn.chunk_words.items()
        if 0 in ws and (1 in ws or 2 in ws) and 3 not in ws
        and 1905 <= syn.chunk_year[cid] <= 1914
    )
    assert got == expected


def test_encode_chunk_context_empty_year_range(codec):
    c, _ = codec
    assert len(c.encode_chunk_context(year_range=(1800, 1850))) == 0


def test_decode_top_words(codec):
    c, _ = codec
    pf = PartialFunc.from_dict({3: 1.0, 7: 5.0, 9: 2.0})
    assert c.decode_top_words(pf, k=2) == [("ord7", 5.0), ("ord9", 2.0)]


def test_decode_chunk_pointers(codec):
    c, syn = codec
    bm = BitMap([(5 << K_SHIFT) | 1, (12 << K_SHIFT) | 0, (40 << K_SHIFT) | 2])
    got = c.decode_chunk_pointers(bm, limit=2)
    assert got == [
        (5, 1, "Bok 5", syn.doc_year[5]),
        (12, 0, "Bok 12", syn.doc_year[12]),
    ]
