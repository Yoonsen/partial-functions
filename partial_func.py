"""
Partielle funksjoner som vektorrom og sannsynlighetsfordelinger over Roaring Bitmaps.

En partiell funksjon f: Words -~-> R representeres uten fikserte N-tupler som:
    dom(f) : pyroaring.BitMap   (en endelig delmengde av ord-ID-er i N)
    val(f) : np.ndarray         (flyttallsarray av lengde |dom(f)|, sortert etter dom(f))

Siden fravær i dom(f) tolkes som 0 under addisjon, oppfyller rommet av partielle
funksjoner med endelig støtte alle aksiomer for et vektorrom over R, uten at
universet av ord trenger å være fiksert på forhånd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
from numba import njit
from pyroaring import BitMap


# ---------------------------------------------------------------------------
# Numba-akselererte kjerner for verdi-innretting (alignment) og POPCNT-blokker
# ---------------------------------------------------------------------------

@njit(cache=True)
def _align_intersection_two_pointer(
    arr_f: np.ndarray,
    val_f: np.ndarray,
    arr_g: np.ndarray,
    val_g: np.ndarray,
    inter_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gitt sorterte domene-arrays arr_f og arr_g og kjent størrelse på snittet,
    plukk ut verdiene vf og vg for snittet i én lineær passering O(|f| + |g|).
    """
    vf = np.empty(inter_len, dtype=np.float64)
    vg = np.empty(inter_len, dtype=np.float64)
    if inter_len == 0:
        return vf, vg

    i = 0
    j = 0
    k = 0
    nf = len(arr_f)
    ng = len(arr_g)
    while i < nf and j < ng:
        wf = arr_f[i]
        wg = arr_g[j]
        if wf == wg:
            vf[k] = val_f[i]
            vg[k] = val_g[j]
            k += 1
            i += 1
            j += 1
        elif wf < wg:
            i += 1
        else:
            j += 1
    return vf, vg


@njit(cache=True)
def _extract_subset_values(
    dom_arr: np.ndarray,
    val_arr: np.ndarray,
    sub_arr: np.ndarray,
) -> np.ndarray:
    """
    Gitt at sub_arr er en delmengde av dom_arr (f.eks. I = dom(f) & dom(g)),
    hent ut val_arr for elementene i sub_arr.
    Bruker galloping / eksponentielt hopp + binærsøk når |sub_arr| << |dom_arr|.
    """
    m = len(sub_arr)
    out = np.empty(m, dtype=np.float64)
    if m == 0:
        return out

    n = len(dom_arr)
    lo = 0
    for k in range(m):
        target = sub_arr[k]
        # Eksponentielt søk (galloping) fra forrige treff lo
        step = 1
        hi = lo
        while hi < n and dom_arr[hi] < target:
            lo = hi + 1
            hi = lo + step
            step <<= 1
        if hi >= n:
            hi = n - 1
        # Binærsøk i [lo, hi]
        while lo <= hi:
            mid = (lo + hi) >> 1
            v = dom_arr[mid]
            if v == target:
                out[k] = val_arr[mid]
                lo = mid + 1
                break
            elif v < target:
                lo = mid + 1
            else:
                hi = mid - 1
    return out


@njit(cache=True)
def _union_add_values(
    arr_f: np.ndarray,
    val_f: np.ndarray,
    arr_g: np.ndarray,
    val_g: np.ndarray,
    alpha: float,
    beta: float,
    union_len: int,
) -> np.ndarray:
    """
    Beregner verdi-arrayet for alpha * f + beta * g over dom(f) | dom(g)
    i ett enkelt flettesveip.
    """
    out = np.empty(union_len, dtype=np.float64)
    i = 0
    j = 0
    k = 0
    nf = len(arr_f)
    ng = len(arr_g)
    while i < nf and j < ng:
        wf = arr_f[i]
        wg = arr_g[j]
        if wf == wg:
            out[k] = alpha * val_f[i] + beta * val_g[j]
            i += 1
            j += 1
        elif wf < wg:
            out[k] = alpha * val_f[i]
            i += 1
        else:
            out[k] = beta * val_g[j]
            j += 1
        k += 1
    while i < nf:
        out[k] = alpha * val_f[i]
        i += 1
        k += 1
    while j < ng:
        out[k] = beta * val_g[j]
        j += 1
        k += 1
    return out


@njit(cache=True)
def _popcnt64(x: np.uint64) -> int:
    """Hardware-vennlig 64-bit popcount (SWAR / LLVM intrinsic via Numba)."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def _build_bitblocks(dom_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Konverterer et sortert domene-array til glisne 64-bits blokker:
      block_ids : uint32[B]  (w >> 6)
      masks     : uint64[B]  (bitmaske for de 64 ordene i blokken)
      offsets   : int32[B]   (start-indeks i val-arrayet, prefikssum av popcnt)
    """
    n = len(dom_arr)
    if n == 0:
        return (
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.int32),
        )
    # Tell antall unike 64-blokker
    n_blocks = 1
    prev_blk = dom_arr[0] >> np.uint32(6)
    for i in range(1, n):
        blk = dom_arr[i] >> np.uint32(6)
        if blk != prev_blk:
            n_blocks += 1
            prev_blk = blk

    block_ids = np.empty(n_blocks, dtype=np.uint32)
    masks = np.zeros(n_blocks, dtype=np.uint64)
    offsets = np.empty(n_blocks, dtype=np.int32)

    b_idx = 0
    curr_blk = dom_arr[0] >> np.uint32(6)
    block_ids[0] = curr_blk
    offsets[0] = 0
    mask = np.uint64(0)

    for i in range(n):
        w = dom_arr[i]
        blk = w >> np.uint32(6)
        if blk != curr_blk:
            masks[b_idx] = mask
            b_idx += 1
            curr_blk = blk
            block_ids[b_idx] = curr_blk
            offsets[b_idx] = i
            mask = np.uint64(0)
        bit = w & np.uint32(63)
        mask |= np.uint64(1) << np.uint64(bit)

    masks[b_idx] = mask
    return block_ids, masks, offsets


@njit(cache=True)
def _bitblock_intersect_dot_and_jsd(
    blk_f: np.ndarray,
    mask_f: np.ndarray,
    off_f: np.ndarray,
    val_f: np.ndarray,
    blk_g: np.ndarray,
    mask_g: np.ndarray,
    off_g: np.ndarray,
    val_g: np.ndarray,
) -> Tuple[float, float, float, int]:
    """
    Demonstrerer direkte bit-nivå snitt med POPCNT-indeksering uten oppslag:
    For hver 64-ords blokk der blk_f[i] == blk_g[j]:
      m_inter = mask_f[i] & mask_g[j]
      For hver satt bit i m_inter:
        lowbit = m_inter & -m_inter
        idx_f  = off_f[i] + popcnt(mask_f[i] & (lowbit - 1))
        idx_g  = off_g[j] + popcnt(mask_g[j] & (lowbit - 1))
    Returnerer (dot, jsd_inter_sum, tv_inter_sum, n_common).
    """
    i = 0
    j = 0
    nf = len(blk_f)
    ng = len(blk_g)
    dot = 0.0
    jsd_sum = 0.0
    tv_sum = 0.0
    n_common = 0

    while i < nf and j < ng:
        bf = blk_f[i]
        bg = blk_g[j]
        if bf == bg:
            mf = mask_f[i]
            mg = mask_g[j]
            m_inter = mf & mg
            if m_inter != np.uint64(0):
                of = off_f[i]
                og = off_g[j]
                while m_inter != np.uint64(0):
                    # Isoler laveste satte bit: lowbit = m_inter & (~m_inter + 1)
                    lowbit = m_inter & (np.uint64(0) - m_inter)
                    before_mask = lowbit - np.uint64(1)
                    idx_f = of + _popcnt64(mf & before_mask)
                    idx_g = og + _popcnt64(mg & before_mask)

                    vf = val_f[idx_f]
                    vg = val_g[idx_g]
                    dot += vf * vg
                    if vf > 0.0 and vg > 0.0:
                        s = vf + vg
                        jsd_sum += vf * np.log2(vf / s) + vg * np.log2(vg / s)
                        tv_sum += vf if vf < vg else vg
                    n_common += 1
                    m_inter ^= lowbit
            i += 1
            j += 1
        elif bf < bg:
            i += 1
        else:
            j += 1

    return dot, jsd_sum, tv_sum, n_common


@njit(cache=True)
def _workspace_dot_and_jsd(
    work: np.ndarray,
    dom_g: np.ndarray,
    val_g: np.ndarray,
) -> Tuple[float, float, float, int]:
    """
    Når f allerede er lagt ut i den flate arbeidsbufferen `work`:
    Ett enkelt sveip over dom(g) henter f(w) = work[w] direkte i O(|g|) tid
    uten søk og uten allokeringer!
    """
    dot = 0.0
    jsd_sum = 0.0
    tv_sum = 0.0
    n_common = 0
    for j in range(len(dom_g)):
        w = dom_g[j]
        vf = work[w]
        if vf != 0.0:
            vg = val_g[j]
            dot += vf * vg
            if vf > 0.0 and vg > 0.0:
                s = vf + vg
                jsd_sum += vf * np.log2(vf / s) + vg * np.log2(vg / s)
                tv_sum += vf if vf < vg else vg
            n_common += 1
    return dot, 1.0 + 0.5 * jsd_sum, 1.0 - tv_sum, n_common


# ---------------------------------------------------------------------------
# Kjerneklasse: PartialFunc (f: Words -~-> Reals)
# ---------------------------------------------------------------------------

class PartialFunc:
    """
    En partiell funksjon f: N -~-> R representert ved:
      - dom : pyroaring.BitMap (delmengde av ord-ID-er)
      - val : np.ndarray[float64] (verdier i eksakt samme rekkefølge som dom)
    """

    __slots__ = ("dom", "val", "_dom_arr_cache", "_bitblocks_cache")

    def __init__(
        self,
        dom: BitMap,
        val: np.ndarray,
        *,
        copy: bool = False,
        validate: bool = True,
    ) -> None:
        if copy:
            self.dom = dom.copy()
            self.val = np.array(val, dtype=np.float64, copy=True)
        else:
            self.dom = dom
            self.val = np.asarray(val, dtype=np.float64)

        if validate and len(self.dom) != len(self.val):
            raise ValueError(
                f"Domene-kardinalitet ({len(self.dom)}) != lengde på verdi-array ({len(self.val)})"
            )
        self._dom_arr_cache: Optional[np.ndarray] = None
        self._bitblocks_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    @classmethod
    def zero(cls) -> PartialFunc:
        """Null-elementet i vektorrommet: den tomme partielle funksjonen."""
        return cls(BitMap(), np.empty(0, dtype=np.float64), validate=False)

    @classmethod
    def from_dict(cls, mapping: Dict[int, float]) -> PartialFunc:
        """Bygger en PartialFunc fra en {word_id: value}-dict."""
        if not mapping:
            return cls.zero()
        items = sorted(mapping.items(), key=lambda kv: kv[0])
        keys = np.array([k for k, _ in items], dtype=np.uint32)
        vals = np.array([v for _, v in items], dtype=np.float64)
        bm = BitMap(keys)
        pf = cls(bm, vals, validate=False)
        pf._dom_arr_cache = keys
        return pf

    @classmethod
    def from_sorted_arrays(cls, word_ids: np.ndarray, values: np.ndarray) -> PartialFunc:
        """Bygger en PartialFunc direkte fra sorterte uint32 ord-ID-er og verdier."""
        keys = np.ascontiguousarray(word_ids, dtype=np.uint32)
        vals = np.ascontiguousarray(values, dtype=np.float64)
        bm = BitMap(keys)
        pf = cls(bm, vals, validate=True)
        pf._dom_arr_cache = keys
        return pf

    @property
    def dom_array(self) -> np.ndarray:
        """Returnerer dom(f) som et sortert uint32 NumPy-array (null-kopi via buffer, cachet)."""
        if self._dom_arr_cache is None:
            self._dom_arr_cache = np.frombuffer(self.dom.to_array(), dtype=np.uint32)
        return self._dom_arr_cache

    @property
    def bitblocks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returnerer (block_ids, masks64, offsets) for direkte POPCNT-indeksering."""
        if self._bitblocks_cache is None:
            self._bitblocks_cache = _build_bitblocks(self.dom_array)
        return self._bitblocks_cache

    def __len__(self) -> int:
        return len(self.dom)

    def __contains__(self, word_id: int) -> bool:
        return word_id in self.dom

    def __getitem__(self, word_id: int) -> float:
        """
        Evaluerer f(w). Returnerer 0.0 hvis w ikke er i dom(f).
        Bruker Roaring sin innebygde O(1)/POPCNT rank(w).
        """
        if word_id not in self.dom:
            return 0.0
        idx = self.dom.rank(word_id) - 1
        return float(self.val[idx])

    def prune_zeros(self, tol: float = 1e-15) -> PartialFunc:
        """Fjerner elementer w der |f(w)| <= tol (kanonisk støtte supp(f))."""
        mask = np.abs(self.val) > tol
        if np.all(mask):
            return self
        kept_ids = self.dom_array[mask]
        kept_vals = self.val[mask]
        return PartialFunc.from_sorted_arrays(kept_ids, kept_vals)

    # -----------------------------------------------------------------------
    # Vektorromsoperasjoner (addisjon over union, skalarmultiplikasjon)
    # -----------------------------------------------------------------------

    def linear_comb(self, other: PartialFunc, alpha: float = 1.0, beta: float = 1.0) -> PartialFunc:
        """Beregner alpha * self + beta * other over dom(self) | dom(other)."""
        if len(self) == 0 and len(other) == 0:
            return PartialFunc.zero()
        if len(self) == 0:
            return PartialFunc(other.dom.copy(), beta * other.val, validate=False)
        if len(other) == 0:
            return PartialFunc(self.dom.copy(), alpha * self.val, validate=False)

        union_bm = self.dom | other.dom
        union_vals = _union_add_values(
            self.dom_array,
            self.val,
            other.dom_array,
            other.val,
            float(alpha),
            float(beta),
            len(union_bm),
        )
        return PartialFunc(union_bm, union_vals, validate=False)

    def __add__(self, other: PartialFunc) -> PartialFunc:
        return self.linear_comb(other, 1.0, 1.0)

    def __sub__(self, other: PartialFunc) -> PartialFunc:
        return self.linear_comb(other, 1.0, -1.0)

    def __neg__(self) -> PartialFunc:
        return PartialFunc(self.dom.copy(), -self.val, validate=False)

    def __mul__(self, other: Union[float, int, PartialFunc]) -> PartialFunc:
        """
        Hvis other er en skalar c: skalarmultiplikasjon (c * f) på dom(f).
        Hvis other er en PartialFunc g: punktvis produkt (f * g) på dom(f) & dom(g).
        """
        if isinstance(other, (int, float, np.number)):
            return PartialFunc(self.dom.copy(), float(other) * self.val, validate=False)
        if isinstance(other, PartialFunc):
            inter_bm, vf, vg = self.align_intersection(other)
            return PartialFunc(inter_bm, vf * vg, validate=False)
        return NotImplemented

    def __rmul__(self, scalar: Union[float, int]) -> PartialFunc:
        if isinstance(scalar, (int, float, np.number)):
            return PartialFunc(self.dom.copy(), float(scalar) * self.val, validate=False)
        return NotImplemented

    def __truediv__(self, other: Union[float, int, PartialFunc]) -> PartialFunc:
        """Skalardivisjon f / c eller punktvis kvotient f / g på dom(f) & dom(g)."""
        if isinstance(other, (int, float, np.number)):
            return PartialFunc(self.dom.copy(), self.val / float(other), validate=False)
        if isinstance(other, PartialFunc):
            inter_bm, vf, vg = self.align_intersection(other)
            return PartialFunc(inter_bm, vf / vg, validate=False)
        return NotImplemented

    # -----------------------------------------------------------------------
    # Restriksjon og verdi-uthenting på snitt: dom(f) & dom(g)
    # -----------------------------------------------------------------------

    def restrict(self, sub_dom: BitMap) -> PartialFunc:
        """
        Restriksjon f|_S til en delmengde S: dom(f|_S) = dom(f) & S.
        Bruker galloping/binærsøk for å hente ut verdiene i S.
        """
        inter_bm = self.dom & sub_dom
        if len(inter_bm) == 0:
            return PartialFunc.zero()
        if len(inter_bm) == len(self.dom):
            return self
        inter_arr = np.frombuffer(inter_bm.to_array(), dtype=np.uint32)
        sub_vals = _extract_subset_values(self.dom_array, self.val, inter_arr)
        pf = PartialFunc(inter_bm, sub_vals, validate=False)
        pf._dom_arr_cache = inter_arr
        return pf

    def align_intersection(
        self,
        other: PartialFunc,
        method: str = "gallop",
    ) -> Tuple[BitMap, np.ndarray, np.ndarray]:
        """
        Finner snittet I = dom(self) & dom(other) med Roaring Bitmap og
        returnerer (I, val_self_on_I, val_other_on_I).

        Metoder:
          - 'gallop'     : Roaring C/SIMD snitt I = dom_f & dom_g + galloping uttrekk.
                           Best når snittet |I| er mye mindre enn |f| eller |g|.
          - 'twopointer' : Roaring snitt + to-peker sveip over dom_f og dom_g.
          - 'rank'       : Roaring sin innebygde bm.rank(w) per element i I.
        """
        inter_bm = self.dom & other.dom
        k = len(inter_bm)
        if k == 0:
            empty = np.empty(0, dtype=np.float64)
            return inter_bm, empty, empty

        if method == "gallop":
            inter_arr = np.frombuffer(inter_bm.to_array(), dtype=np.uint32)
            vf = _extract_subset_values(self.dom_array, self.val, inter_arr)
            vg = _extract_subset_values(other.dom_array, other.val, inter_arr)
            return inter_bm, vf, vg

        elif method == "twopointer":
            vf, vg = _align_intersection_two_pointer(
                self.dom_array, self.val, other.dom_array, other.val, k
            )
            return inter_bm, vf, vg

        elif method == "rank":
            f_dom = self.dom
            g_dom = other.dom
            idx_f = np.fromiter((f_dom.rank(w) - 1 for w in inter_bm), dtype=np.int64, count=k)
            idx_g = np.fromiter((g_dom.rank(w) - 1 for w in inter_bm), dtype=np.int64, count=k)
            return inter_bm, self.val[idx_f], other.val[idx_g]

        else:
            raise ValueError(f"Ukjent metode: {method}")

    def dot(self, other: PartialFunc) -> float:
        """Indreprodukt <f, g> = sum_{w in dom(f) & dom(g)} f(w) * g(w)."""
        if self.dom.isdisjoint(other.dom):
            return 0.0
        _, vf, vg = self.align_intersection(other, method="gallop")
        return float(np.dot(vf, vg))

    def __matmul__(self, other: PartialFunc) -> float:
        return self.dot(other)

    # -----------------------------------------------------------------------
    # Sannsynlighetsfordelinger, PMI og informasjonsteoretiske distanser
    # -----------------------------------------------------------------------

    def l1_norm(self) -> float:
        return float(np.sum(np.abs(self.val)))

    def l2_norm(self) -> float:
        return float(np.linalg.norm(self.val))

    def to_prob(self) -> PartialFunc:
        """Normaliserer en ikke-negativ partiell funksjon til en sannsynlighetsfordeling (sum = 1)."""
        total = float(np.sum(self.val))
        if total <= 0.0:
            return PartialFunc.zero()
        return PartialFunc(self.dom.copy(), self.val / total, validate=False)

    def pmi(
        self,
        bg_prob: PartialFunc,
        *,
        alpha: float = 1.0,
        shift_k: float = 1.0,
        positive: bool = True,
    ) -> PartialFunc:
        """
        Beregner (Shifted) PMI eller PPMI mot en bakgrunnsfordeling bg_prob:
            PMI(w) = log2( P_f(w) / P_bg(w)^alpha ) - log2(shift_k)
        Definert på dom(f) & dom(bg_prob).
        """
        pf = self.to_prob()
        inter_bm, p_f, p_bg = pf.align_intersection(bg_prob, method="gallop")
        if len(inter_bm) == 0:
            return PartialFunc.zero()

        scores = np.log2(p_f / (p_bg ** alpha)) - np.log2(shift_k)
        if positive:
            keep = scores > 0.0
            if not np.all(keep):
                inter_arr = np.frombuffer(inter_bm.to_array(), dtype=np.uint32)[keep]
                return PartialFunc.from_sorted_arrays(inter_arr, scores[keep])
        return PartialFunc(inter_bm, scores, validate=False)

    def jsd(self, other: PartialFunc) -> float:
        """
        Jensen-Shannon Divergens JSD(P || Q) i bits (mellom 0 og 1),
        beregnet KUN over snittet I = dom(P) & dom(Q)!

        Fordi P og Q summerer til 1 over sine respektive domener, gjelder identiteten:
          JSD(P || Q) = 1 + 0.5 * sum_{w in dom(P) & dom(Q)} [
              P(w) * log2(P(w) / (P(w) + Q(w))) + Q(w) * log2(Q(w) / (P(w) + Q(w)))
          ]
        Vi trenger altså ALDRI konstruere unionen dom(P) | dom(Q)!
        """
        if len(self) == 0 or len(other) == 0:
            return 1.0
        if self.dom.isdisjoint(other.dom):
            return 1.0
        _, p, q = self.align_intersection(other, method="gallop")
        mask = (p > 0.0) & (q > 0.0)
        p = p[mask]
        q = q[mask]
        s = p + q
        inter_term = np.sum(p * np.log2(p / s) + q * np.log2(q / s))
        return float(max(0.0, min(1.0, 1.0 + 0.5 * inter_term)))

    def total_variation(self, other: PartialFunc) -> float:
        """
        Total Variation distanse TV(P, Q) = 0.5 * ||P - Q||_1 in [0, 1],
        beregnet KUN over snittet I = dom(P) & dom(Q):
          TV(P, Q) = 1 - sum_{w in dom(P) & dom(Q)} min(P(w), Q(w))
        """
        if self.dom.isdisjoint(other.dom):
            return 1.0
        _, p, q = self.align_intersection(other, method="gallop")
        return float(max(0.0, 1.0 - np.sum(np.minimum(p, q))))

    def hellinger(self, other: PartialFunc) -> float:
        """
        Hellinger-distanse H(P, Q) in [0, 1], beregnet KUN over snittet dom(P) & dom(Q):
          H(P, Q) = sqrt(1 - sum_{w in dom(P) & dom(Q)} sqrt(P(w) * Q(w)))
        """
        if self.dom.isdisjoint(other.dom):
            return 1.0
        _, p, q = self.align_intersection(other, method="gallop")
        bc = float(np.sum(np.sqrt(np.maximum(0.0, p * q))))
        return float(np.sqrt(max(0.0, 1.0 - bc)))

    # -----------------------------------------------------------------------
    # Ettersporbarhet (Traceability): Dekomponering av skalarer til PartialFunc
    # -----------------------------------------------------------------------

    def explain_dot(self, other: PartialFunc) -> PartialFunc:
        """
        I stedet for å redusere <f, g> til ett tall, returneres den partielle
        funksjonen h = f * g over dom(f) & dom(g) der sum(h) == <f, g>.
        Hvert element h(w) viser eksakt hvor mye ord w bidrar til indreproduktet.
        """
        return self * other

    def explain_overlap(self, other: PartialFunc) -> PartialFunc:
        """
        For to sannsynlighetsfordelinger P og Q: returnerer den partielle
        funksjonen m(w) = min(P(w), Q(w)) på dom(P) & dom(Q).
        Summen av m(w) er 1 - TV(P, Q), og hvert element viser ordets eksakte
        bidrag til den delte sannsynlighetsmassen.
        """
        inter_bm, p, q = self.align_intersection(other, method="gallop")
        return PartialFunc(inter_bm, np.minimum(p, q), validate=False)

    def explain_jsd_divergence(self, other: PartialFunc) -> PartialFunc:
        """
        Dekomponerer Jensen-Shannon Divergens JSD(P || Q) i det eksakte,
        additive bidraget fra hvert enkelt ord w in dom(P) | dom(Q):
            d_JSD(w) = 0.5 * [ P(w) log2(2 P(w) / (P(w)+Q(w)))
                             + Q(w) log2(2 Q(w) / (P(w)+Q(w))) ]
        slik at sum_{w in dom(P) | dom(Q)} d_JSD(w) == JSD(P || Q).
        Ord som kun finnes i P (Q(w)=0) bidrar med eksakt 0.5 * P(w).
        """
        # Start med 0.5 * P + 0.5 * Q på unionen
        base = self.linear_comb(other, 0.5, 0.5)
        if not self.dom.isdisjoint(other.dom):
            inter_bm, p, q = self.align_intersection(other, method="gallop")
            mask = (p > 0.0) & (q > 0.0)
            s = p[mask] + q[mask]
            # Korrigering på snittet: legg til 0.5 * (p log2(p/s) + q log2(q/s))
            corr_vals = 0.5 * (p[mask] * np.log2(p[mask] / s) + q[mask] * np.log2(q[mask] / s))
            inter_arr = np.frombuffer(inter_bm.to_array(), dtype=np.uint32)[mask]
            corr_pf = PartialFunc.from_sorted_arrays(inter_arr, corr_vals)
            return base + corr_pf
        return base

    def top_k(self, k: int = 10, *, largest: bool = True) -> List[Tuple[int, float]]:
        """
        Plukker ut de k største (eller minste) (id, verdi)-parene i funksjonen.
        Brukes i dekodingsfasen slik at man kun dekoder de ID-ene som trengs.
        """
        if len(self) == 0 or k <= 0:
            return []
        k = min(k, len(self))
        if largest:
            idx = np.argpartition(self.val, -k)[-k:]
            idx = idx[np.argsort(self.val[idx])[::-1]]
        else:
            idx = np.argpartition(self.val, k - 1)[:k]
            idx = idx[np.argsort(self.val[idx])]
        dom_arr = self.dom_array
        return [(int(dom_arr[i]), float(self.val[i])) for i in idx]


# ---------------------------------------------------------------------------
# Sparse Accumulator Workspace (Gustavsons triks for O(|I|) uten søk)
# ---------------------------------------------------------------------------

class SparseWorkspace:
    """
    Gjenbrukbar flat arbeidsbuffer ('Sparse Accumulator' / Gustavsons algoritme).
    For et univers på f.eks. 2 millioner ord tar denne kun 16 MB (float64) eller 8 MB (float32).
    Gjør at snitt, PMI mot bakgrunn, prikkprodukter og JSD mot mange partielle
    funksjoner skjer ved direkte array-indeksering work[I] uten søk!
    """

    def __init__(self, universe_size: int) -> None:
        self.universe_size = universe_size
        self.work = np.zeros(universe_size, dtype=np.float64)
        self._active_dom: Optional[ np.ndarray ] = None

    def scatter(self, f: PartialFunc) -> None:
        """Legger f sine verdier direkte ut på sine ord-ID-posisjoner i O(|f|)."""
        if self._active_dom is not None:
            self.work[self._active_dom] = 0.0
        dom_arr = f.dom_array
        self.work[dom_arr] = f.val
        self._active_dom = dom_arr

    def clear(self) -> None:
        """Nullstiller kun de posisjonene som sist ble satt (O(|f|))."""
        if self._active_dom is not None:
            self.work[self._active_dom] = 0.0
            self._active_dom = None

    def gather_on_bitmap(self, bm: BitMap) -> np.ndarray:
        """Henter verdiene for en hvilken som helst bitmap I direkte i O(|I|)."""
        return self.work[bm.to_array()]

    def dot_with_active(self, g: PartialFunc) -> float:
        """
        Når f allerede er scattered i work:
        <f, g> = work[dom(g)] @ val(g)
        Trenger ikke engang regne ut snittet dom(f) & dom(g) fordi work er 0 utenfor dom(f)!
        """
        return float(np.dot(self.work[g.dom_array], g.val))

    def dot_and_jsd_with_active(self, g: PartialFunc) -> Tuple[float, float, float, int]:
        """Returnerer (dot, jsd, tv, n_common) via Numba O(|g|) oppslag i work."""
        return _workspace_dot_and_jsd(self.work, g.dom_array, g.val)
