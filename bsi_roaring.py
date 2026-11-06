"""
Bit-Sliced Indexing (BSI) for Partielle Funksjoner: 100 % Roaring Bitmaps.

I stedet for å lagre verdiene i et separat flyttalls-array (som krever rank/oppslag
ved snitt), representeres både domenet OG verdiene som 32-bits Roaring Bitmaps:

  - dom       : BitMap (alle aktive ord w i funksjonen)
  - slices[b] : BitMap (alle ord w der bit nr. b i heltallsverdien v(w) er 1)
  - scale     : float  (slik at reell verdi er f(w) = v(w) / scale)

Alle operasjoner reduseres dermed til ren mengde-algebra på Roaring Bitmaps:
  1. Restriksjon til snitt S:  slices[b] & S   (0 oppslag!)
  2. Sum over delmengde S:     sum_b 2^b * |slices[b] & S|
  3. Vektoraddisjon f + g:     Ripple-carry full-adder med XOR (^) og AND (&)
  4. Terskelsøk (f >= T):      Top-down bitplan-filtrering i O(B) bit-operasjoner
  5. Indreprodukt <f, g>:      sum_{i,j} 2^(i+j) * |F_i & G_j|
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from pyroaring import BitMap

from partial_func import PartialFunc


class BSIPartialFunc:
    """
    Partiell funksjon f: Words -~-> R representert 100 % som en stabel av
    32-bits Roaring Bitmaps (Bit-Sliced Index).
    """

    __slots__ = ("dom", "slices", "scale")

    def __init__(
        self,
        dom: BitMap,
        slices: Sequence[BitMap],
        scale: float = 1.0,
    ) -> None:
        self.dom = dom
        self.slices: Tuple[BitMap, ...] = tuple(slices)
        self.scale = float(scale)

    @property
    def n_bits(self) -> int:
        return len(self.slices)

    def __len__(self) -> int:
        return len(self.dom)

    @classmethod
    def from_partial_func(
        cls,
        pf: PartialFunc,
        *,
        n_bits: int = 8,
        max_val: Optional[float] = None,
        scale: Optional[float] = None,
    ) -> BSIPartialFunc:
        """
        Konverterer en vanlig PartialFunc til en Bit-Sliced Roaring-funksjon
        med `n_bits` bitplan (f.eks. 8 bit -> 0..255, eller 12 bit -> 0..4095).
        """
        if len(pf) == 0:
            return cls(BitMap(), (), scale=1.0)

        dom_arr = pf.dom_array
        vals = np.maximum(0.0, pf.val)
        max_int = (1 << n_bits) - 1

        if scale is None:
            mv = float(max_val) if max_val is not None else float(np.max(vals))
            scale = max_int / mv if mv > 0.0 else 1.0

        int_vals = np.clip(np.rint(vals * scale), 0, max_int).astype(np.uint32)

        slices: List[BitMap] = []
        for b in range(n_bits):
            mask = ((int_vals >> b) & 1) != 0
            slices.append(BitMap(dom_arr[mask]))

        return cls(pf.dom.copy(), slices, scale=scale)

    def to_partial_func(self) -> PartialFunc:
        """Dekoder bit-planene tilbake til en vanlig PartialFunc(dom, val)."""
        if len(self.dom) == 0:
            return PartialFunc.zero()
        dom_arr = np.frombuffer(self.dom.to_array(), dtype=np.uint32)
        acc = np.zeros(len(dom_arr), dtype=np.float64)
        for b, s_bm in enumerate(self.slices):
            if len(s_bm) == 0:
                continue
            s_arr = np.frombuffer(s_bm.to_array(), dtype=np.uint32)
            idx = np.searchsorted(dom_arr, s_arr)
            acc[idx] += float(1 << b)
        return PartialFunc(self.dom.copy(), acc / self.scale, validate=False)

    def __getitem__(self, word_id: int) -> float:
        """Evaluerer f(w) ved å sjekke hvilke bit-plan som inneholder w."""
        if word_id not in self.dom:
            return 0.0
        v = 0
        for b, s_bm in enumerate(self.slices):
            if word_id in s_bm:
                v |= 1 << b
        return v / self.scale

    def serialized_bytes(self) -> int:
        """Totalt antall bytes for dom + alle bit-plan."""
        return len(self.dom.serialize()) + sum(len(s.serialize()) for s in self.slices)

    # -----------------------------------------------------------------------
    # 1. Restriksjon til snitt / delmengde: 0 oppslag!
    # -----------------------------------------------------------------------

    def restrict(self, sub_dom: BitMap) -> BSIPartialFunc:
        """
        Restriksjon f|_S til en delmengde S:
        Kun bitvis AND mellom hvert bit-plan og S! Ingen rank eller søk!
        """
        new_dom = self.dom & sub_dom
        if len(new_dom) == 0:
            return BSIPartialFunc(new_dom, (), scale=self.scale)
        new_slices = [s & new_dom for s in self.slices]
        return BSIPartialFunc(new_dom, new_slices, scale=self.scale)

    # -----------------------------------------------------------------------
    # 2. Sum over hele domenet eller et snitt S: kun POPCNT!
    # -----------------------------------------------------------------------

    def sum_all(self) -> float:
        """Eksakt sum av alle verdier i f via kardinaliteten til hvert bit-plan."""
        total_int = sum((1 << b) * len(s) for b, s in enumerate(self.slices))
        return total_int / self.scale

    def sum_on(self, sub_dom: BitMap) -> float:
        """
        Eksakt sum_{w in sub_dom} f(w) uten å hente ut et eneste enkeltord:
            sum_{b=0}^{B-1} 2^b * |slices[b] & sub_dom|
        """
        total_int = sum(
            (1 << b) * s.intersection_cardinality(sub_dom)
            for b, s in enumerate(self.slices)
        )
        return total_int / self.scale

    # -----------------------------------------------------------------------
    # 3. Vektoraddisjon (f + g) som Ripple-Carry Full-Adder i Roaring!
    # -----------------------------------------------------------------------

    def __add__(self, other: BSIPartialFunc) -> BSIPartialFunc:
        """
        Adderer to BSI-funksjoner med samme skala over hele ordrommet samtidig
        ved hjelp av en binær full-adder (XOR og AND mellom Roaring Bitmaps!).
        """
        if not np.isclose(self.scale, other.scale):
            raise ValueError("BSI-addisjon krever samme scale på begge funksjoner.")

        max_b = max(self.n_bits, other.n_bits)
        carry = BitMap()
        out_slices: List[BitMap] = []

        for b in range(max_b):
            a_bm = self.slices[b] if b < self.n_bits else BitMap()
            b_bm = other.slices[b] if b < other.n_bits else BitMap()

            axb = a_bm ^ b_bm
            sum_bm = axb ^ carry
            carry = (a_bm & b_bm) | (axb & carry)
            out_slices.append(sum_bm)

        if len(carry) > 0:
            out_slices.append(carry)

        return BSIPartialFunc(self.dom | other.dom, out_slices, scale=self.scale)

    # -----------------------------------------------------------------------
    # 4. Terskelsøk: finn alle ord der f(w) >= threshold i O(B) bit-operasjoner
    # -----------------------------------------------------------------------

    def threshold_bitmap(self, min_value: float) -> BitMap:
        """
        Returnerer bitmappet {w in dom(f) | f(w) >= min_value} utelukkende
        ved top-down bitplan-logikk (ingen løkke over ordene!).
        """
        target_int = int(np.ceil(min_value * self.scale))
        if target_int <= 0:
            return self.dom.copy()
        if target_int >= (1 << self.n_bits):
            return BitMap()

        gt = BitMap()             # Ord som allerede er strengt større enn target
        eq = self.dom.copy()      # Ord som har samme prefiks som target så langt

        for b in range(self.n_bits - 1, -1, -1):
            s_bm = self.slices[b]
            bit_required = (target_int >> b) & 1
            if bit_required == 1:
                # For å holde tritt med target må bit b være 1
                eq &= s_bm
            else:
                # Target har 0 i bit b -> alle i eq som har 1 i bit b er nå strengt større!
                gt |= eq & s_bm
                eq -= s_bm

        return gt | eq

    # -----------------------------------------------------------------------
    # 5. Indreprodukt <f, g> rent via intersection_cardinality mellom bit-plan
    # -----------------------------------------------------------------------

    def dot(self, other: BSIPartialFunc) -> float:
        """
        Beregner <f, g> = sum_{w in dom(f) & dom(g)} f(w) * g(w)
        utelukkende med Roaring intersection_cardinality:
            <f, g> = (1 / (scale_f * scale_g)) * sum_{i, j} 2^(i+j) * |F_i & G_j|
        """
        inter_dom = self.dom & other.dom
        if len(inter_dom) == 0:
            return 0.0

        # Restriker ikke-tomme bitplan til snittet inter_dom først
        f_active = [
            (1 << i, s & inter_dom)
            for i, s in enumerate(self.slices)
            if not s.isdisjoint(inter_dom)
        ]
        g_active = [
            (1 << j, s & inter_dom)
            for j, s in enumerate(other.slices)
            if not s.isdisjoint(inter_dom)
        ]

        total_int = 0
        for wi, bm_i in f_active:
            if len(bm_i) == 0:
                continue
            for wj, bm_j in g_active:
                c = bm_i.intersection_cardinality(bm_j)
                if c > 0:
                    total_int += (wi * wj) * c

        return total_int / (self.scale * other.scale)


if __name__ == "__main__":
    from test_mat_and_totals import load_universe_and_books

    word_to_id, id_to_word, bg_totals, bg_prob, books_meta, book_funcs = (
        load_universe_and_books()
    )

    print("=" * 76)
    print("TEST & BENCHMARK: Bit-Sliced Roaring (BSIPartialFunc) vs. PartialFunc (SoA)")
    print("=" * 76)

    # 1. Konverter de 338 kokebøkene til:
    #    a) Eksakte råfrekvens-BSI (10 bitplan, scale=1.0 -> eksakte heltall 0..1023!)
    #    b) Kvantiserte PPMI-BSI (8 bitplan, scale=10.0 -> oppløsning 0.1 i PPMI!)
    t0 = time.perf_counter()
    bsi_freq_books = [
        BSIPartialFunc.from_partial_func(f, n_bits=10, scale=1.0)
        for f in book_funcs
    ]
    ppmi_books = [
        f.pmi(bg_prob, alpha=0.90, shift_k=2.0, positive=True)
        for f in book_funcs
    ]
    bsi_ppmi_8bit = [
        BSIPartialFunc.from_partial_func(p, n_bits=8, scale=10.0)
        for p in ppmi_books
    ]
    bsi_ppmi_6bit = [
        BSIPartialFunc.from_partial_func(p, n_bits=6, scale=3.0)
        for p in ppmi_books
    ]
    build_ms = (time.perf_counter() - t0) * 1000

    # Verifiser eksakt heltalls-aritmetikk på bsi_freq_books[0] og [1]
    f0, f1 = book_funcs[0], book_funcs[1]
    bf0, bf1 = bsi_freq_books[0], bsi_freq_books[1]
    assert np.isclose(f0.l1_norm(), bf0.sum_all()), "Sum feilet!"
    assert np.isclose((f0 + f1).l1_norm(), (bf0 + bf1).sum_all()), "Ripple-carry addisjon feilet!"
    decoded_sum = (bf0 + bf1).to_partial_func()
    assert len(((f0 + f1) - decoded_sum).prune_zeros()) == 0, "Eksakt full-adder feilet!"

    # Verifiser terskelsøk (f0 >= 20) i O(B) bit-operasjoner
    bm_ge20_bsi = bf0.threshold_bitmap(20.0)
    bm_ge20_ref = BitMap(f0.dom_array[f0.val >= 20.0])
    assert bm_ge20_bsi == bm_ge20_ref, "Terskelsøk feilet!"

    # Minneforbruk per bok
    soa_bytes = np.mean([len(p.dom.serialize()) + p.val.astype(np.float32).nbytes for p in ppmi_books])
    bsi8_bytes = np.mean([b.serialized_bytes() for b in bsi_ppmi_8bit])
    bsi6_bytes = np.mean([b.serialized_bytes() for b in bsi_ppmi_6bit])

    print(f"[1] Verifikasjon av ren bitplan-algebra (eksakte frekvenser, 10 bitplan):")
    print(f"    - Ripple-carry addisjon (bf0 + bf1) == (f0 + f1) : Eksakt identisk!")
    print(f"    - Terskelsøk bf0.threshold_bitmap(20)            : {len(bm_ge20_bsi)} ord (eksakt match!)")
    print(f"\n[2] Minnefotavtrykk per bok (snitt over 338 bøker, PPMI-vektor):")
    print(f"    - PartialFunc (dom: BitMap32 + float32[]) : {soa_bytes:6.0f} bytes / bok")
    print(f"    - BSIPartialFunc (8 bitplan, 0.1-steg)    : {bsi8_bytes:6.0f} bytes / bok")
    print(f"    - BSIPartialFunc (6 bitplan, 0.33-steg)   : {bsi6_bytes:6.0f} bytes / bok")

    # Benchmark av de 4 operasjonene der BSI og PartialFunc skiller lag:
    sub_mask = BitMap(range(1, 5000))  # F.eks. en ordgalakse / delmengde av ord

    # A. Restriksjon til snitt S: f.restrict(S)
    t0 = time.perf_counter()
    for _ in range(10):
        for p in ppmi_books:
            _ = p.restrict(sub_mask)
    t_soa_restrict = (time.perf_counter() - t0) / (10 * len(ppmi_books)) * 1e6

    t0 = time.perf_counter()
    for _ in range(10):
        for b in bsi_ppmi_6bit:
            _ = b.restrict(sub_mask)
    t_bsi_restrict = (time.perf_counter() - t0) / (10 * len(bsi_ppmi_6bit)) * 1e6

    # B. Sum av vekter over en ordmengde/galakse S: sum_{w in S} f(w)
    t0 = time.perf_counter()
    for _ in range(10):
        for p in ppmi_books:
            _ = np.sum(p.restrict(sub_mask).val)
    t_soa_sum_on = (time.perf_counter() - t0) / (10 * len(ppmi_books)) * 1e6

    t0 = time.perf_counter()
    for _ in range(10):
        for b in bsi_ppmi_6bit:
            _ = b.sum_on(sub_mask)
    t_bsi_sum_on = (time.perf_counter() - t0) / (10 * len(bsi_ppmi_6bit)) * 1e6

    # C. Terskelfilter: finn bitmappet av ord med PPMI >= 5.0
    t0 = time.perf_counter()
    for _ in range(10):
        for p in ppmi_books:
            _ = BitMap(p.dom_array[p.val >= 5.0])
    t_soa_thresh = (time.perf_counter() - t0) / (10 * len(ppmi_books)) * 1e6

    t0 = time.perf_counter()
    for _ in range(10):
        for b in bsi_ppmi_6bit:
            _ = b.threshold_bitmap(5.0)
    t_bsi_thresh = (time.perf_counter() - t0) / (10 * len(bsi_ppmi_6bit)) * 1e6

    # D. Parvis indreprodukt <f, g> + korrelasjon med eksakt float64 indreprodukt
    anchor_soa = ppmi_books[0]
    anchor_bsi8 = bsi_ppmi_8bit[0]
    anchor_bsi6 = bsi_ppmi_6bit[0]

    t0 = time.perf_counter()
    dots_soa = [anchor_soa.dot(p) for p in ppmi_books[1:]]
    t_soa_dot = (time.perf_counter() - t0) / (len(ppmi_books) - 1) * 1e6

    t0 = time.perf_counter()
    dots_bsi8 = [anchor_bsi8.dot(b) for b in bsi_ppmi_8bit[1:]]
    t_bsi8_dot = (time.perf_counter() - t0) / (len(bsi_ppmi_8bit) - 1) * 1e6

    t0 = time.perf_counter()
    dots_bsi6 = [anchor_bsi6.dot(b) for b in bsi_ppmi_6bit[1:]]
    t_bsi6_dot = (time.perf_counter() - t0) / (len(bsi_ppmi_6bit) - 1) * 1e6

    corr8 = float(np.corrcoef(dots_soa, dots_bsi8)[0, 1])
    corr6 = float(np.corrcoef(dots_soa, dots_bsi6)[0, 1])

    print(f"\n[3] Hastighets-sammenligning (per bok / bokpar):")
    print(f"    Operasjon                           | PartialFunc (BitMap+float) | BSI-Roaring (6 bitplan)")
    print(f"    ------------------------------------+----------------------------+------------------------")
    print(f"    Restriksjon til delmengde f|_S      | {t_soa_restrict:18.2f} µs      | {t_bsi_restrict:16.2f} µs")
    print(f"    Sum over ordgalakse sum_{{w in S}} f  | {t_soa_sum_on:18.2f} µs      | {t_bsi_sum_on:16.2f} µs")
    print(f"    Terskelsøk {{w | f(w) >= 5.0}}        | {t_soa_thresh:18.2f} µs      | {t_bsi_thresh:16.2f} µs")
    print(f"    Parvis indreprodukt <f, g>          | {t_soa_dot:18.2f} µs      | {t_bsi6_dot:16.2f} µs (r={corr6:.5f})")
    print(f"    Parvis indreprodukt <f, g> (8-bit)  |                            | {t_bsi8_dot:16.2f} µs (r={corr8:.6f})")
