"""
Koding -> Funksjons-algebra -> Dekoding (med 100 % ettersporbarhet).

Demonstrerer arbeidsflyten:
1. KODING (Encode):
   Tekstlige spørringer, ordlister eller tidsintervaller oversettes én gang ved
   inngangen til heltalls-bitmaps (pyroaring.BitMap) og partielle funksjoner (PartialFunc).
2. FUNKSJONS-ALGEBRA (Pure Algebra):
   Alle beregninger (snitt, union, PMI, vektordifferanser, JSD, prikkprodukter,
   og dekomponering av likhet/divergens) utføres utelukkende på (BitMap, ndarray)
   uten strenger og uten tabelloppslag.
3. DEKODING & ETTERSPORBARHET (Decode & Trace):
   Kun de få elementene som faktisk skal vises (topp-k ord, hvilke eksakte ord som
   forklarer likheten mellom to tekster, eller hvilke eksakte bok-chunks `(doc_id, chunk)`
   som skal hentes ut til konkordans) dekodes ved utgangen.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
from pyroaring import BitMap

from partial_func import PartialFunc
from collocations_and_trends import (
    K_SHIFT,
    RoaringCorpusIndex,
    build_roaring_index_from_bundle,
)
from test_mat_and_totals import MAT_BUNDLE_PATH, load_universe_and_books


class AlgebraCodec:
    """
    Koder ord og betingelser inn i Roaring/PartialFunc-algebraen, og dekoder
    kun topp-resultater eller konkordans-koordinater tilbake til lesbar form.
    """

    def __init__(
        self,
        idx: RoaringCorpusIndex,
        books_meta: List[dict],
        book_funcs: List[PartialFunc],
        bg_prob: PartialFunc,
    ) -> None:
        self.idx = idx
        self.books_meta = books_meta
        self.book_funcs = book_funcs
        self.bg_prob = bg_prob

    # --- 1. INN-KODING (Fra strenger/vilkår til BitMap & PartialFunc) ---

    def encode_word_id(self, word: str) -> int:
        wid = self.idx.word_to_id.get(word.lower())
        if wid is None:
            raise KeyError(f"Ukjent ord: {word}")
        return wid

    def encode_words_bitmap(self, words: Iterable[str]) -> BitMap:
        return BitMap(
            self.idx.word_to_id[w.lower()]
            for w in words
            if w.lower() in self.idx.word_to_id
        )

    def encode_chunk_context(
        self,
        *,
        all_of: Sequence[str] = (),
        any_of: Sequence[str] = (),
        none_of: Sequence[str] = (),
        year_range: Tuple[int, int] | None = None,
    ) -> BitMap:
        """
        Koder en sammensatt kontekst-spørring direkte til én Roaring chunk-bitmap A.
        """
        bm = self.idx.all_chunks_bm.copy()
        if year_range is not None:
            y0, y1 = year_range
            yr_bms = [
                b for y, b in self.idx.year_chunk_bm.items() if y0 <= y <= y1
            ]
            bm &= BitMap.union(*yr_bms) if yr_bms else BitMap()

        for w in all_of:
            bm &= self.idx.word_chunks(w)
        if any_of:
            or_bms = [self.idx.word_chunks(w) for w in any_of]
            bm &= BitMap.union(*or_bms)
        for w in none_of:
            bm -= self.idx.word_chunks(w)
        return bm

    # --- 3. UT-DEKODING (Kun for topp-k resultater og konkordans-pekere) ---

    def decode_top_words(
        self, pf: PartialFunc, k: int = 8, *, largest: bool = True
    ) -> List[Tuple[str, float]]:
        return [
            (self.idx.id_to_word[wid], val)
            for wid, val in pf.top_k(k, largest=largest)
        ]

    def decode_chunk_pointers(
        self, chunk_bm: BitMap, limit: int = 5
    ) -> List[Tuple[int, int, str, int]]:
        """
        Dekoder kun de første `limit` chunk-ID-ene tilbake til
        (doc_id, local_chunk, boktittel, år) for uthenting av konkordans.
        """
        out = []
        for idx_i, c_id in enumerate(chunk_bm):
            if idx_i >= limit:
                break
            doc_id = c_id >> K_SHIFT
            local_chunk = c_id & ((1 << K_SHIFT) - 1)
            meta = self.books_meta[doc_id]
            out.append((doc_id, local_chunk, meta["title"], int(meta["year"])))
        return out


if __name__ == "__main__":
    word_to_id, id_to_word, bg_totals, bg_prob, books_meta, book_funcs = (
        load_universe_and_books()
    )
    idx = build_roaring_index_from_bundle(MAT_BUNDLE_PATH, word_to_id, id_to_word)
    codec = AlgebraCodec(idx, books_meta, book_funcs, bg_prob)

    print("=" * 76)
    print("DEMO: Koding -> Funksjons-algebra -> Ettersporbar Dekoding")
    print("=" * 76)

    # 1. KODING: Vi koder to bøker (eller to kollokasjoner) inn i algebraen
    b0, b1 = 0, 14  # Bok 0 (1890) og Bok 14 (1989: Lærebog i Huusholdningen)
    f0 = book_funcs[b0].pmi(bg_prob, alpha=0.90, shift_k=2.0)
    f1 = book_funcs[b1].pmi(bg_prob, alpha=0.90, shift_k=2.0)
    # Normaliser til L2-enhetsvektorer i PartialFunc-rommet
    u0 = f0 / f0.l2_norm()
    u1 = f1 / f1.l2_norm()

    # 2. ALGEBRA: Cosinus-likhet er bare summen av den partielle funksjonen h = u0 * u1
    cos_sim = u0 @ u1
    explanation_pf = u0.explain_dot(u1)  # PartialFunc over dom(u0) & dom(u1)!
    assert np.isclose(cos_sim, np.sum(explanation_pf.val))

    # 3. DEKODING: Hvilke eksakte ord står for mesteparten av likheten cos_sim?
    top_contrib = codec.decode_top_words(explanation_pf, k=8)
    top8_share = sum(v for _, v in top_contrib) / cos_sim * 100.0

    print(
        f"\n[1] Ettersporbar likhet mellom Bok #{b0} ({books_meta[b0]['year']}) og Bok #{b1} ({books_meta[b1]['year']}):\n"
        f"    Cosinus-likhet (u0 @ u1) = {cos_sim:.4f} over |dom(u0) & dom(u1)| = {len(explanation_pf)} felles ord.\n"
        f"    Topp 8 ord står alene for {top8_share:.1f}% av likheten (u0 * u1):"
    )
    for w, contrib in top_contrib:
        print(f"      - {w:14s}: +{contrib:.4f} ({100 * contrib / cos_sim:4.1f}% av total likhet)")

    # Dekomponering av JSD-divergens: Hvilke ord skiller bøkene mest fra hverandre?
    p0 = book_funcs[b0].to_prob()
    p1 = book_funcs[b1].to_prob()
    jsd_total = p0.jsd(p1)
    jsd_decomp_pf = p0.explain_jsd_divergence(p1)
    assert np.isclose(jsd_total, np.sum(jsd_decomp_pf.val))

    top_div_words = codec.decode_top_words(jsd_decomp_pf, k=6)
    print(
        f"\n[2] Ettersporbar JSD-divergens: JSD(P0 || P1) = {jsd_total:.4f} bits (eksakt sum av {len(jsd_decomp_pf)} ord-bidrag)\n"
        f"    Ordene som bidrar mest til forskjellen mellom bøkene:"
    )
    for w, bits in top_div_words:
        wid = codec.encode_word_id(w)
        print(
            f"      - {w:14s}: +{bits:.4f} bits  "
            f"(P0={100*p0[wid]:.2f}%, P1={100*p1[wid]:.2f}%)"
        )

    # 4. Fra funksjons-algebra helt tilbake til konkordans-kontekst (kun ved behov!)
    anchor_bm = codec.encode_chunk_context(
        all_of=["hvitløk", "olivenolje"],
        year_range=(1990, 2022),
    )
    _, pmi_colloc = idx.collocation_pmi(
        anchor_bm,
        min_cooc=3,
        exclude_ids=codec.encode_words_bitmap(["hvitløk", "olivenolje"]),
    )
    top_colloc_word, top_pmi_score = codec.decode_top_words(pmi_colloc, k=1)[0]

    # Finn eksakt hvilke tekstblokker (doc_id, local_chunk) som har (hvitløk & olivenolje & topp-kollokat)
    evidence_chunks = anchor_bm & idx.word_chunks(top_colloc_word)
    pointers = codec.decode_chunk_pointers(evidence_chunks, limit=3)

    print(
        f"\n[3] Fra algebra tilbake til tekst-kontekst (konkordans-pekere):\n"
        f"    Spørring: «hvitløk» & «olivenolje» (1990–2022) -> {len(anchor_bm)} chunks\n"
        f"    Sterkeste kollokat i algebraen: «{top_colloc_word}» (PMI={top_pmi_score:.2f}, {len(evidence_chunks)} bevis-chunks)\n"
        f"    Dekodede (doc_id, chunk_id)-pekere klar for tekstuthenting:"
    )
    for doc_id, loc_c, title, yr in pointers:
        print(f"      -> doc_id={doc_id:3d}, chunk #{loc_c:3d} | ({yr}) {title[:55]}")
