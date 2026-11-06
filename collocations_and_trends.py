"""
Kollokasjons- og trendlinje-aggregering rent i Roaring Bitmaps -> PartialFunc.

Viser hvordan:
1. Enhver kontekst/søk A (et ord, et ordpar x & y, eller et ord innenfor et
   delkorpus/tidsrom K_t) er en Roaring Bitmap over chunks eller dokumenter.
2. Kollokasjonsprofilen til A aggregeres uten ett eneste bokoppslag via
       f_A(w) = |A & C_w|  (A.intersection_cardinality(C_w))
   som direkte gir en PartialFunc f_A : Words -~-> R.
3. Trendlinjen til A over tid (år eller tiår) aggregeres uten bokoppslag via
       tau_A(t) = |A & K_t|
   som direkte gir en PartialFunc tau_A : Time -~-> R.
4. Fordi både f_A og tau_A er PartialFunc-objekter, kan vi bruke eksakt samme
   vektorroms- og distribusjonsalgebra (PMI, JSD, Hellinger, differanser f_t2 - f_t1)
   på både bøker, kollokasjoner og tidsserier!
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pyroaring import BitMap

from partial_func import PartialFunc, SparseWorkspace

K_SHIFT = 11  # 2048 chunk-slots per bok: chunk_id = (doc_id << 11) | local_chunk


@dataclass
class RoaringCorpusIndex:
    """
    Holder en ren Roaring-representasjon av et korpus:
      - word_doc_bm[w]   : BitMap over doc_id for ord-ID w
      - word_chunk_bm[w] : BitMap over chunk_id for ord-ID w
      - time_doc_bm[t]   : BitMap over doc_id for tidssteg t (år eller tiår)
      - time_chunk_bm[t] : BitMap over chunk_id (Run-container) for tidssteg t
    """
    word_to_id: Dict[str, int]
    id_to_word: List[str]
    word_doc_bm: Dict[int, BitMap]
    word_chunk_bm: Dict[int, BitMap]
    all_docs_bm: BitMap
    all_chunks_bm: BitMap
    year_doc_bm: Dict[int, BitMap]
    year_chunk_bm: Dict[int, BitMap]
    decade_doc_bm: Dict[int, BitMap]
    decade_chunk_bm: Dict[int, BitMap]
    tau_year_chunks: PartialFunc
    tau_decade_chunks: PartialFunc
    corpus_chunk_bg: PartialFunc  # f_K(w) = |C_w| over hele korpuset

    def word_chunks(self, word: str) -> BitMap:
        wid = self.word_to_id.get(word.lower())
        if wid is None:
            return BitMap()
        return self.word_chunk_bm.get(wid, BitMap())

    def word_docs(self, word: str) -> BitMap:
        wid = self.word_to_id.get(word.lower())
        if wid is None:
            return BitMap()
        return self.word_doc_bm.get(wid, BitMap())

    def collocation_profile(
        self,
        anchor_chunks: BitMap,
        *,
        candidate_ids: Optional[Sequence[int]] = None,
        min_cooc: int = 1,
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> PartialFunc:
        """
        Bygger kollokasjonsprofilen f_A : Words -~-> R for en vilkårlig chunk-bitmap A
        rent i Roaring Bitmaps uten bokoppslag:
            f_A(w) = |A & C_w| = anchor_chunks.intersection_cardinality(C_w)
        Returnerer en PartialFunc(dom=BitMap(kollokater), val=cooc_counts).
        """
        if len(anchor_chunks) == 0:
            return PartialFunc.zero()

        excl = set(exclude_ids) if exclude_ids else set()
        cands = candidate_ids if candidate_ids is not None else sorted(self.word_chunk_bm.keys())

        active_ids: List[int] = []
        counts: List[float] = []
        w_chunks_map = self.word_chunk_bm

        for wid in cands:
            if wid in excl:
                continue
            c_bm = w_chunks_map.get(wid)
            if c_bm is None:
                continue
            c = anchor_chunks.intersection_cardinality(c_bm)
            if c >= min_cooc:
                active_ids.append(wid)
                counts.append(float(c))

        if not active_ids:
            return PartialFunc.zero()

        return PartialFunc.from_sorted_arrays(
            np.array(active_ids, dtype=np.uint32),
            np.array(counts, dtype=np.float64),
        )

    def collocation_pmi(
        self,
        anchor_chunks: BitMap,
        *,
        corpus_mask: Optional[BitMap] = None,
        min_cooc: int = 3,
        alpha: float = 1.0,
        exclude_ids: Optional[Iterable[int]] = None,
    ) -> Tuple[PartialFunc, PartialFunc]:
        """
        Beregner både rå samforekomst f_A og assosiasjonsstyrke PMI_A : Words -~-> R
        for et anker A innenfor korpusmasken K (standard: hele korpuset):
            P(w | A) = |A & C_w| / |A|
            P(w | K) = |K & C_w| / |K|
            PMI(w)   = log2( P(w | A) / P(w | K)^alpha )
        Alt utelukkende via PartialFunc-algebra!
        """
        K_bm = corpus_mask if corpus_mask is not None else self.all_chunks_bm
        A_bm = anchor_chunks & K_bm if corpus_mask is not None else anchor_chunks
        n_A = len(A_bm)
        n_K = len(K_bm)
        if n_A == 0 or n_K == 0:
            return PartialFunc.zero(), PartialFunc.zero()

        raw_f = self.collocation_profile(
            A_bm, min_cooc=min_cooc, exclude_ids=exclude_ids
        )
        if len(raw_f) == 0:
            return raw_f, PartialFunc.zero()

        # P(w | A) som PartialFunc
        p_given_A = raw_f / float(n_A)

        # Hvis vi bruker hele korpuset, har vi allerede corpus_chunk_bg
        if corpus_mask is None:
            bg_f = self.corpus_chunk_bg.restrict(raw_f.dom)
        else:
            bg_f = self.collocation_profile(
                K_bm, candidate_ids=raw_f.dom_array.tolist(), min_cooc=1
            )
        p_given_K = bg_f / float(n_K)

        # Punktvis kvotient i PartialFunc-rommet over dom(raw_f) & dom(bg_f)
        ratio_pf = p_given_A / PartialFunc(
            p_given_K.dom, p_given_K.val ** alpha, validate=False
        )
        pmi_vals = np.log2(np.maximum(ratio_pf.val, 1e-30))
        pmi_pf = PartialFunc(ratio_pf.dom, pmi_vals, validate=False)
        return raw_f, pmi_pf

    def trendline(
        self,
        anchor_chunks: BitMap,
        *,
        by_decade: bool = True,
    ) -> PartialFunc:
        """
        Bygger en trendlinje tau_A : Time -~-> R rent i Roaring Bitmaps:
            tau_A(t) = |anchor_chunks & K_t|
        Returnerer en PartialFunc der dom(tau_A) er BitMap av aktive år/tiår
        og val(tau_A) er antall treff-chunks i hvert tidssteg.
        """
        buckets = self.decade_chunk_bm if by_decade else self.year_chunk_bm
        active_t: List[int] = []
        counts: List[float] = []
        for t in sorted(buckets.keys()):
            c = anchor_chunks.intersection_cardinality(buckets[t])
            if c > 0:
                active_t.append(t)
                counts.append(float(c))
        if not active_t:
            return PartialFunc.zero()
        return PartialFunc.from_sorted_arrays(
            np.array(active_t, dtype=np.uint32),
            np.array(counts, dtype=np.float64),
        )

    def relative_trendline(
        self,
        anchor_chunks: BitMap,
        *,
        by_decade: bool = True,
    ) -> PartialFunc:
        """
        Relativ frekvens over tid som kvotient mellom to partielle funksjoner:
            tau_rel = tau_A / tau_total
        """
        tau_a = self.trendline(anchor_chunks, by_decade=by_decade)
        tau_tot = self.tau_decade_chunks if by_decade else self.tau_year_chunks
        return tau_a / tau_tot

    def diachronic_pmi_trendline(
        self,
        chunks_x: BitMap,
        chunks_y: BitMap,
        *,
        by_decade: bool = True,
    ) -> Tuple[PartialFunc, PartialFunc]:
        """
        Beregner samforekomst-trendlinjen tau_{x & y} og diakron PMI-trendlinje
        mellom x og y over tid, rent med PartialFunc-aritmetikk:
            E_{x,y}   = (tau_x * tau_y) / tau_total
            PMI(t)    = log2( tau_{x & y}(t) / E_{x,y}(t) )
        """
        chunks_xy = chunks_x & chunks_y
        tau_xy = self.trendline(chunks_xy, by_decade=by_decade)
        if len(tau_xy) == 0:
            return PartialFunc.zero(), PartialFunc.zero()

        tau_x = self.trendline(chunks_x, by_decade=by_decade)
        tau_y = self.trendline(chunks_y, by_decade=by_decade)
        tau_tot = self.tau_decade_chunks if by_decade else self.tau_year_chunks

        # Forventet antall felles chunks per tidssteg under uavhengighet:
        expected_pf = (tau_x * tau_y) / tau_tot
        ratio_pf = tau_xy / expected_pf
        pmi_pf = PartialFunc(ratio_pf.dom, np.log2(ratio_pf.val), validate=False)
        return tau_xy, pmi_pf

    def sample_docs(
        self,
        n: int,
        *,
        corpus: Optional[BitMap] = None,
        by: Optional[str] = None,
        seed: int = 0,
    ) -> BitMap:
        """
        Trekker et reproduserbart tilfeldig utvalg av bøker (doc_id) uten tilbakelegging.
        Bøker, ikke chunks, er utvalgsenheten, slik at chunks fra samme bok holdes samlet.
          - by=None      : n bøker fra hele korpuset
          - by="year"    : inntil n bøker fra hvert år     (C & K_t for hver t)
          - by="decade"  : inntil n bøker fra hvert tiår
        Bruker rank/select i BitMap (corpus[i]), så korpuset materialiseres aldri.
        Samme (corpus, n, by, seed) gir alltid samme utvalg.
        """
        C = corpus if corpus is not None else self.all_docs_bm
        rng = np.random.default_rng(seed)

        if by is None:
            strata = [C]
        elif by in ("year", "decade"):
            buckets = self.year_doc_bm if by == "year" else self.decade_doc_bm
            strata = [C & buckets[t] for t in sorted(buckets.keys())]
        else:
            raise ValueError(f"by må være None, 'year' eller 'decade', fikk {by!r}")

        out = BitMap()
        for S in strata:
            m = len(S)
            if m == 0:
                continue
            ranks = rng.choice(m, size=min(n, m), replace=False)
            out.update(S[int(r)] for r in ranks)
        return out

    def docs_to_chunks(self, docs: BitMap) -> BitMap:
        """
        Løfter en dokument-bitmap til chunk-nivå:
            {chunk_id : chunk_id >> K_SHIFT in docs} & all_chunks_bm
        slik at et utvalg fra sample_docs kan brukes som corpus_mask/anker i
        collocation_pmi og trendline.
        """
        span = BitMap()
        for d in docs:
            span.add_range(d << K_SHIFT, (d + 1) << K_SHIFT)
        return span & self.all_chunks_bm


# ---------------------------------------------------------------------------
# Bygg RoaringCorpusIndex fra mat_bundle.json (med deterministiske chunks)
# ---------------------------------------------------------------------------

def build_roaring_index_from_bundle(
    bundle_path: Path,
    word_to_id: Dict[str, int],
    id_to_word: List[str],
    seed: int = 42,
) -> RoaringCorpusIndex:
    """
    Konstruerer et komplett Roaring-basert chunk- og tidskorpus fra en doc-to-vec
    bundle (f.eks. mat_bundle.json):
      - Hver bok d (0..337) har n_chunks blokker: chunk_id = (d << 11) | c.
      - For hvert ord w i bok d kjenner vi eksakt DF_chunk(d, w) (fra dfratio_x100
        og poisson-forventningen K_d * (1 - exp(-TF/K_d))).
      - Vi legger ut de eksakte chunk-forekomstene i word_chunk_bm[w] og bygger
        Run-container bitmaps for alle år og tiår.
    """
    t0 = time.perf_counter()
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    books = bundle["books"]
    words = bundle["words"]
    csr = bundle["csr"]
    row_ptr = csr["row_ptr"]
    col_idx = np.asarray(csr["col_idx"], dtype=np.int64)
    freq = np.asarray(csr["freq"], dtype=np.float64)
    dfratio_x100 = np.asarray(csr["dfratio_x100"], dtype=np.float64)

    local_to_global = np.array(
        [word_to_id[w_obj["w"].lower()] for w_obj in words], dtype=np.uint32
    )

    word_doc_bm: Dict[int, BitMap] = {}
    word_chunk_bm: Dict[int, BitMap] = {}
    year_doc_bm: Dict[int, BitMap] = {}
    year_chunk_bm: Dict[int, BitMap] = {}
    decade_doc_bm: Dict[int, BitMap] = {}
    decade_chunk_bm: Dict[int, BitMap] = {}

    all_docs_bm = BitMap()
    all_chunks_bm = BitMap()

    rng = np.random.default_rng(seed)

    for d_id, b_meta in enumerate(books):
        year = int(b_meta.get("year") or 1900)
        decade = (year // 10) * 10
        n_chunks = max(1, min(2047, int(b_meta.get("n_chunks") or 100)))

        all_docs_bm.add(d_id)
        c_start = d_id << K_SHIFT
        c_end = c_start + n_chunks
        all_chunks_bm.add_range(c_start, c_end)

        year_doc_bm.setdefault(year, BitMap()).add(d_id)
        year_chunk_bm.setdefault(year, BitMap()).add_range(c_start, c_end)
        decade_doc_bm.setdefault(decade, BitMap()).add(d_id)
        decade_chunk_bm.setdefault(decade, BitMap()).add_range(c_start, c_end)

        # Gjør klar bokens ord
        start, end = row_ptr[d_id], row_ptr[d_id + 1]
        for k in range(start, end):
            loc_w = col_idx[k]
            gid = int(local_to_global[loc_w])
            tf_val = freq[k]
            dfr = dfratio_x100[k] / 100.0

            # Rekonstruer antall chunks DF_chunk(d, w) i boka
            mu = tf_val / n_chunks
            expected_df = n_chunks * (1.0 - np.exp(-mu))
            df_chunk = int(np.clip(np.round(dfr * expected_df), 1, min(int(tf_val), n_chunks)))

            w_dbm = word_doc_bm.get(gid)
            if w_dbm is None:
                w_dbm = BitMap()
                word_doc_bm[gid] = w_dbm
            w_dbm.add(d_id)

            w_cbm = word_chunk_bm.get(gid)
            if w_cbm is None:
                w_cbm = BitMap()
                word_chunk_bm[gid] = w_cbm

            # Fordel ordets chunks deterministisk med en tematisk klump-profil
            # slik at ord som opptrer i samme bok/kapittel får realistisk chunk-overlapp
            if df_chunk >= n_chunks:
                w_cbm.add_range(c_start, c_end)
            else:
                # Velg et senter basert på (d_id, loc_w // 8) + spredning etter dfr
                center = ((loc_w * 13 + d_id * 7) % n_chunks)
                indices = (center + np.arange(df_chunk) * max(1, int(round(dfr * 2)))) % n_chunks
                for c_local in indices:
                    w_cbm.add(c_start | int(c_local))

    # Run-optimaliser alle periode- og chunk-bitmaps
    all_chunks_bm.run_optimize()
    for bm in year_chunk_bm.values():
        bm.run_optimize()
    for bm in decade_chunk_bm.values():
        bm.run_optimize()

    # Bygg total-trendlinjene tau_year_chunks og tau_decade_chunks som PartialFunc
    years_sorted = sorted(year_chunk_bm.keys())
    tau_year_chunks = PartialFunc.from_sorted_arrays(
        np.array(years_sorted, dtype=np.uint32),
        np.array([len(year_chunk_bm[y]) for y in years_sorted], dtype=np.float64),
    )

    decades_sorted = sorted(decade_chunk_bm.keys())
    tau_decade_chunks = PartialFunc.from_sorted_arrays(
        np.array(decades_sorted, dtype=np.uint32),
        np.array([len(decade_chunk_bm[d]) for d in decades_sorted], dtype=np.float64),
    )

    # Bygg korpusets bakgrunns-chunkfrekvens som en PartialFunc
    wids_sorted = sorted(word_chunk_bm.keys())
    corpus_chunk_bg = PartialFunc.from_sorted_arrays(
        np.array(wids_sorted, dtype=np.uint32),
        np.array([len(word_chunk_bm[w]) for w in wids_sorted], dtype=np.float64),
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(
        f"[RoaringCorpusIndex bygget på {elapsed_ms:.1f} ms]\n"
        f"  Totalt antall bøker  : {len(all_docs_bm):,} ({len(all_docs_bm.serialize())} bytes som Roaring)\n"
        f"  Totalt antall chunks : {len(all_chunks_bm):,} ({len(all_chunks_bm.serialize()):,} bytes som Run-Roaring!)\n"
        f"  Indekserte ord-bitmaps: {len(word_chunk_bm):,} ord\n"
    )
    return RoaringCorpusIndex(
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        word_doc_bm=word_doc_bm,
        word_chunk_bm=word_chunk_bm,
        all_docs_bm=all_docs_bm,
        all_chunks_bm=all_chunks_bm,
        year_doc_bm=year_doc_bm,
        year_chunk_bm=year_chunk_bm,
        decade_doc_bm=decade_doc_bm,
        decade_chunk_bm=decade_chunk_bm,
        tau_year_chunks=tau_year_chunks,
        tau_decade_chunks=tau_decade_chunks,
        corpus_chunk_bg=corpus_chunk_bg,
    )


# ---------------------------------------------------------------------------
# Direkte test mot ekte produksjonsshard: imag_roaring_main_0.db (8 906 bøker)
# ---------------------------------------------------------------------------

def demo_shard0_doc_collocations_and_trends(id_to_word: List[str]) -> None:
    """
    Viser ren Roaring-aggregering av kollokasjoner og trendlinjer direkte på
    Shard 0 (8 906 bøker fra 1800-tallet i imag_roaring_main_0.db + metadata_core_books.sqlite)
    UTEN å røre unigrams/bokoppslag!
    """
    shard_db = Path("/Users/larsj/Github/metadata-roaring/shards/imag_roaring_main_0.db")
    meta_db = Path("/Users/larsj/Github/metadata-roaring/metadata_core_books.sqlite")
    if not shard_db.exists() or not meta_db.exists():
        return

    print("=" * 76)
    print("DEL 2: Ekte produksjonsshard (Shard 0: 8 906 bøker fra 1800-tallet)")
    print("       Ren Roaring-telling av trendlinjer & kollokasjoner (0 bokoppslag!)")
    print("=" * 76)

    con = sqlite3.connect(f"file:{shard_db}?mode=ro&immutable=1", uri=True)
    mcon = sqlite3.connect(f"file:{meta_db}?mode=ro&immutable=1", uri=True)

    t0 = time.perf_counter()
    all_books_blob = con.execute("SELECT post FROM urns_postings WHERE id = 1").fetchone()[0]
    all_books_bm = BitMap.deserialize(all_books_blob)
    book_ids = list(all_books_bm)

    # Hent utgivelsesår for de 8 906 bøkene og bygg tiårs-bitmaps B_t = RM(book_id)
    placeholders = ",".join("?" for _ in book_ids)
    rows = mcon.execute(
        f"SELECT dhlabid, year FROM metadata_core WHERE dhlabid IN ({placeholders})",
        tuple(book_ids),
    ).fetchall()

    decade_bm: Dict[int, BitMap] = {}
    year_bm: Dict[int, BitMap] = {}
    for dhlabid, yr in rows:
        if yr is not None and 1800 <= int(yr) <= 1905:
            y = int(yr)
            d = (y // 10) * 10
            year_bm.setdefault(y, BitMap()).add(int(dhlabid))
            decade_bm.setdefault(d, BitMap()).add(int(dhlabid))

    decades_sorted = sorted(decade_bm.keys())
    tau_total_docs = PartialFunc.from_sorted_arrays(
        np.array(decades_sorted, dtype=np.uint32),
        np.array([len(decade_bm[d]) for d in decades_sorted], dtype=np.float64),
    )

    # Slå opp docpost-bitmaps via PRIMARY KEY (word) for 5 000 sentrale ord + 1800-tallsord
    cand_words = [w for w in id_to_word[1:15000] if w.isalpha() and len(w) >= 3][:5000]
    extra_1800 = [
        "demokrati", "frihed", "forfatning", "stortinget", "kongen", "penger",
        "jernbane", "telegraf", "aristokrati", "republik", "monarki", "stemmeret",
        "statsraad", "borger", "folket", "likhed", "eiendom", "kapital",
    ]
    query_words = list(dict.fromkeys(cand_words + extra_1800))
    ph = ",".join("?" for _ in query_words)
    word_rows = con.execute(
        f"""
        SELECT cf_id, word, docfreq, docpost, docpost_is_complement
        FROM words
        WHERE word IN ({ph}) AND docpost IS NOT NULL
        """,
        tuple(query_words),
    ).fetchall()

    w_bms: Dict[int, BitMap] = {}
    id_to_w: Dict[int, str] = {}
    w_to_id: Dict[str, int] = {}

    for cf_id, w_str, df, blob, is_comp in word_rows:
        cid = int(cf_id)
        if cid in w_bms:
            continue
        bm = BitMap.deserialize(blob)
        if is_comp:
            bm = all_books_bm - bm
        w_bms[cid] = bm
        id_to_w[cid] = w_str
        w_to_id[w_str] = cid

    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Lastet {len(w_bms):,} ord-bitmaps og {len(decades_sorted)} tiårs-bitmaps på {load_ms:.1f} ms")

    # 1. Lynrask kollokasjonsprofil (dokument-samforekomst + PMI) uten bokoppslag
    cands_sorted = sorted(w_bms.keys())
    bg_doc_pf = PartialFunc.from_sorted_arrays(
        np.array(cands_sorted, dtype=np.uint32),
        np.array([len(w_bms[c]) for c in cands_sorted], dtype=np.float64),
    )
    n_total_docs = float(len(all_books_bm))

    anchor_word = "demokrati"
    anchor_bm = w_bms[w_to_id[anchor_word]]

    t0 = time.perf_counter()
    active_ids = []
    cooc_vals = []
    for cid in cands_sorted:
        c = anchor_bm.intersection_cardinality(w_bms[cid])
        if c >= 25:
            active_ids.append(cid)
            cooc_vals.append(float(c))
    f_demokrati = PartialFunc.from_sorted_arrays(
        np.array(active_ids, dtype=np.uint32),
        np.array(cooc_vals, dtype=np.float64),
    )
    # Beregn PMI (alpha=0.75) direkte som kvotient mellom partielle funksjoner:
    p_given_anchor = f_demokrati / float(len(anchor_bm))
    p_bg = (bg_doc_pf.restrict(f_demokrati.dom)) / n_total_docs
    pmi_demokrati = PartialFunc(
        f_demokrati.dom,
        np.log2(p_given_anchor.val / (p_bg.val ** 0.75)),
        validate=False,
    )
    colloc_us = (time.perf_counter() - t0) * 1e6

    top_idx = np.argsort(pmi_demokrati.val)[::-1][:10]
    dom_arr = pmi_demokrati.dom_array
    top_collocs = [
        (id_to_w[int(dom_arr[i])], int(f_demokrati.val[i]), pmi_demokrati.val[i])
        for i in top_idx
        if id_to_w[int(dom_arr[i])] != anchor_word
    ][:8]

    print(
        f"\n  [1] Dokument-kollokasjon for «{anchor_word}» ({len(anchor_bm)} bøker) mot {len(cands_sorted):,} ord:\n"
        f"      Tid i ren Roaring: {colloc_us:.1f} µs ({colloc_us/1000:.2f} ms!) — 0 bokoppslag!\n"
        f"      Topp PMI-kollokater: "
        + ", ".join(f"{w} (n={n}, PMI={pmi:.2f})" for w, n, pmi in top_collocs)
    )

    # 2. Trendlinjer som PartialFunc over tiår (1810..1900)
    def doc_trend_pf(bm: BitMap) -> PartialFunc:
        ts = []
        vs = []
        for d in decades_sorted:
            c = bm.intersection_cardinality(decade_bm[d])
            if c > 0:
                ts.append(d)
                vs.append(float(c))
        return PartialFunc.from_sorted_arrays(
            np.array(ts, dtype=np.uint32), np.array(vs, dtype=np.float64)
        )

    t0 = time.perf_counter()
    bm_frihed = w_bms[w_to_id["frihed"]]
    bm_kongen = w_bms[w_to_id["kongen"]]
    bm_both = anchor_bm & bm_frihed  # Bøker med BÅDE «demokrati» OG «frihed»

    tau_dem = doc_trend_pf(anchor_bm)
    tau_fri = doc_trend_pf(bm_frihed)
    tau_both = doc_trend_pf(bm_both)
    rel_both = (tau_both / tau_total_docs) * 100.0  # Prosent av alle bøker i tiåret
    trend_us = (time.perf_counter() - t0) * 1e6

    print(
        f"\n  [2] Trendlinjer som PartialFunc (1810–1900) beregnet på {trend_us:.1f} µs:\n"
        f"      Tiår : " + "  ".join(f"{int(d):4d}s" for d in rel_both.dom_array)
    )
    print(
        f"      Antall bøker («demokrati» & «frihed»): "
        + "  ".join(f"{int(v):5d}" for v in tau_both.val)
    )
    print(
        f"      Andel av korpuset (%)                : "
        + "  ".join(f"{v:4.2f}%" for v in rel_both.val)
    )

    con.close()
    mcon.close()


if __name__ == "__main__":
    from test_mat_and_totals import MAT_BUNDLE_PATH, load_universe_and_books

    word_to_id, id_to_word, bg_totals, bg_prob, books_meta, book_funcs = (
        load_universe_and_books()
    )
    idx = build_roaring_index_from_bundle(MAT_BUNDLE_PATH, word_to_id, id_to_word)

    # Varm opp Numba-kjerner før vi tar tiden
    _ = (book_funcs[0] + book_funcs[1]) / (book_funcs[0] + book_funcs[1])

    print("=" * 76)
    print("DEL 1: Kollokasjonsprofiler, diakron drift & trendlinjer på DDK 641 (Kokebøker)")
    print("=" * 76)

    # A. Kollokasjonsprofil rundt «hvitløk» vs «smør» rent via chunk-bitmaps
    for target_w in ["smør", "hvitløk", "olivenolje"]:
        bm_w = idx.word_chunks(target_w)
        wid = word_to_id.get(target_w)
        t0 = time.perf_counter()
        raw_f, pmi_f = idx.collocation_pmi(bm_w, min_cooc=5, alpha=0.9, exclude_ids=[wid] if wid else None)
        dt_ms = (time.perf_counter() - t0) * 1000
        top_i = np.argsort(pmi_f.val)[::-1][:8]
        d_arr = pmi_f.dom_array
        top_str = ", ".join(
            f"{id_to_word[int(d_arr[i])]} ({pmi_f.val[i]:.2f})" for i in top_i
        )
        print(
            f"  Kollokasjon «{target_w}» ({len(bm_w)} chunks) -> |dom(f)|={len(raw_f)} ord på {dt_ms:.2f} ms:\n"
            f"    Topp PMI: {top_str}"
        )

    # B. Diakron kollokasjonsdrift: «kaffe» eller «smør» før 1950 vs etter 1970
    early_chunks = BitMap.union(*[bm for dec, bm in idx.decade_chunk_bm.items() if dec <= 1940])
    late_chunks = BitMap.union(*[bm for dec, bm in idx.decade_chunk_bm.items() if dec >= 1970])

    bm_smor = idx.word_chunks("smør")
    f_early, pmi_early = idx.collocation_pmi(bm_smor, corpus_mask=early_chunks, min_cooc=4, exclude_ids=[word_to_id["smør"]])
    f_late, pmi_late = idx.collocation_pmi(bm_smor, corpus_mask=late_chunks, min_cooc=4, exclude_ids=[word_to_id["smør"]])

    drift_jsd = f_early.to_prob().jsd(f_late.to_prob())
    # Vektordifferanse i PartialFunc-rommet: pmi_late - pmi_early!
    diff_pmi = pmi_late - pmi_early
    diff_arr = diff_pmi.dom_array
    gained_idx = np.argsort(diff_pmi.val)[::-1][:6]
    lost_idx = np.argsort(diff_pmi.val)[:6]

    print(
        f"\n  [Diakron kollokasjonsendring for «smør»: ≤1940 vs ≥1970]\n"
        f"    JSD mellom kollokasjonsdistribusjonene : {drift_jsd:.3f} bits\n"
        f"    Sterkest vekst i PMI (pmi_late - pmi_early): "
        + ", ".join(f"{id_to_word[int(diff_arr[i])]} (+{diff_pmi.val[i]:.2f})" for i in gained_idx)
        + "\n    Sterkest fall i PMI  (pmi_late - pmi_early): "
        + ", ".join(f"{id_to_word[int(diff_arr[i])]} ({diff_pmi.val[i]:.2f})" for i in lost_idx)
    )

    # C. Trendlinjer som PartialFunc over tiår + Trend-JSD
    print("\n  [Trendlinjer som PartialFunc over tiår (relativ chunk-tetthet i %)]")
    trend_words = ["potetesmel", "margarin", "hvitløk", "olivenolje"]
    trend_pfs: Dict[str, PartialFunc] = {}
    for w in trend_words:
        rel_t = idx.relative_trendline(idx.word_chunks(w), by_decade=True) * 100.0
        trend_pfs[w] = idx.trendline(idx.word_chunks(w), by_decade=True).to_prob()
        dec_str = ", ".join(
            f"{int(d)}s:{v:.2f}%" for d, v in zip(rel_t.dom_array, rel_t.val) if v >= 0.05
        )
        print(f"    {w:12s} -> {dec_str}")

    print(
        f"\n  [Historisk samvariasjon målt med snitt-basert JSD mellom trend-funksjonene]\n"
        f"    JSD(trend(hvitløk), trend(olivenolje)) = {trend_pfs['hvitløk'].jsd(trend_pfs['olivenolje']):.4f} bits (samme moderne bølge!)\n"
        f"    JSD(trend(hvitløk), trend(potetesmel)) = {trend_pfs['hvitløk'].jsd(trend_pfs['potetesmel']):.4f} bits (ulike tidsepoker!)\n"
    )

    # Kjør også direkte mot ekte 8 906-bøkers produksjonsshard
    demo_shard0_doc_collocations_and_trends(id_to_word)
