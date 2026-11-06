"""
Native NEAR_k over rank-indeksert word_unigrams («arena»-modellen), kompilert med Numba.

For hver kandidatbok b klippes posisjons-bitmapene ut av pos_arena med
    r = docs.rank(b) - 1,  blob = pos_arena[offsets[r] : offsets[r + 1]]
og dekodes direkte fra Roarings portable serialiseringsformat til sorterte
uint32-posisjoner, uten pyroaring og uten Python per bok. NEAR telles så med
et glidende vindu over de to sorterte listene.

Semantikk som api_python.near_bitmap_from_group_bitmaps i sqlite-backend:
    p i A er et treff hvis det finnes q i B med |p - q| <= window,
    og q != p når A og B er samme ord (exclude_self).
Resultatet er (antall treffposisjoner, antall bøker med treff).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit
from pyroaring import BitMap

SERIAL_COOKIE_NO_RUN = 12346
SERIAL_COOKIE = 12347
NO_OFFSET_THRESHOLD = 4
ARRAY_MAX = 4096


# Alle heltall holdes som int64: i Numba gir uint64 blandet med int64 float64.
@njit(cache=True, inline="always")
def _u16(buf: np.ndarray, i: int) -> int:
    return np.int64(buf[i]) | (np.int64(buf[i + 1]) << 8)


@njit(cache=True, inline="always")
def _u32(buf: np.ndarray, i: int) -> int:
    return (np.int64(buf[i]) | (np.int64(buf[i + 1]) << 8)
            | (np.int64(buf[i + 2]) << 16) | (np.int64(buf[i + 3]) << 24))


@njit(cache=True)
def roaring_decode(buf: np.ndarray, start: int, end: int, out: np.ndarray) -> int:
    """
    Dekoder én portable-serialisert 32-bits Roaring-bitmap i buf[start:end]
    til sorterte uint32-verdier i out. Returnerer antall verdier, eller -1
    hvis out er for liten.
    """
    p = start
    cookie = _u32(buf, p)
    p += 4
    has_run = (cookie & 0xFFFF) == SERIAL_COOKIE
    if has_run:
        size = (cookie >> 16) + 1
        run_flags = p
        p += (size + 7) // 8
    elif cookie == SERIAL_COOKIE_NO_RUN:
        size = _u32(buf, p)
        p += 4
        run_flags = -1
    else:
        return -2

    header = p
    p += 4 * size
    if (not has_run) or size >= NO_OFFSET_THRESHOLD:
        p += 4 * size  # offset-tabellen trengs ikke ved sekvensiell lesing

    n = 0
    for c in range(size):
        key = _u16(buf, header + 4 * c) << 16
        card = _u16(buf, header + 4 * c + 2) + 1
        is_run = has_run and ((np.int64(buf[run_flags + c // 8]) >> (c % 8)) & 1) == 1
        if n + card > out.shape[0]:
            return -1
        if is_run:
            n_runs = _u16(buf, p)
            p += 2
            for r in range(n_runs):
                v0 = _u16(buf, p)
                ln = _u16(buf, p + 2)
                p += 4
                for v in range(v0, v0 + ln + 1):
                    out[n] = key | v
                    n += 1
        elif card <= ARRAY_MAX:
            for i in range(card):
                out[n] = key | _u16(buf, p)
                n += 1
                p += 2
        else:
            for byte_i in range(8192):
                byte = np.int64(buf[p + byte_i])
                if byte:
                    for bit in range(8):
                        if (byte >> bit) & 1:
                            out[n] = key | (byte_i * 8 + bit)
                            n += 1
            p += 8192
    return n


@njit(cache=True)
def _near_count(a: np.ndarray, na: int, b: np.ndarray, nb: int, window: int, exclude_self: bool) -> int:
    hits = 0
    j = 0
    for i in range(na):
        p = np.int64(a[i])
        while j < nb and np.int64(b[j]) < p - window:
            j += 1
        k = j
        while k < nb and np.int64(b[k]) <= p + window:
            if not (exclude_self and np.int64(b[k]) == p):
                hits += 1
                break
            k += 1
    return hits


# ---------------------------------------------------------------------------
# Tett vei: bitsett over hele boka for ord der posisjonene er tette
# (f.eks. «.» og «,»), slik at NEAR blir shift/OR/AND/POPCNT over 64-bits ord
# i stedet for dekoding og fletting av millioner av heltall.
# ---------------------------------------------------------------------------

@njit(cache=True)
def roaring_card_span(buf: np.ndarray, start: int) -> Tuple[int, int]:
    """Leser bare headeren: (kardinalitet, antall 64-bits ord som dekker største nøkkel)."""
    cookie = _u32(buf, start)
    p = start + 4
    if (cookie & 0xFFFF) == SERIAL_COOKIE:
        size = (cookie >> 16) + 1
        p += (size + 7) // 8
    elif cookie == SERIAL_COOKIE_NO_RUN:
        size = _u32(buf, p)
        p += 4
    else:
        return -2, 0
    if size == 0:
        return 0, 0
    card = 0
    for c in range(size):
        card += _u16(buf, p + 4 * c + 2) + 1
    max_key = _u16(buf, p + 4 * (size - 1))
    return card, (max_key + 1) * 1024


@njit(cache=True)
def roaring_to_bitset(buf: np.ndarray, start: int, bits: np.ndarray) -> int:
    """Skriver posisjonene inn i bits (uint64, nullstilt på forhånd). Returnerer kardinalitet."""
    p = start
    cookie = _u32(buf, p)
    p += 4
    has_run = (cookie & 0xFFFF) == SERIAL_COOKIE
    if has_run:
        size = (cookie >> 16) + 1
        run_flags = p
        p += (size + 7) // 8
    elif cookie == SERIAL_COOKIE_NO_RUN:
        size = _u32(buf, p)
        p += 4
        run_flags = -1
    else:
        return -2
    header = p
    p += 4 * size
    if (not has_run) or size >= NO_OFFSET_THRESHOLD:
        p += 4 * size

    n = 0
    one = np.uint64(1)
    for c in range(size):
        word0 = _u16(buf, header + 4 * c) * 1024
        card = _u16(buf, header + 4 * c + 2) + 1
        is_run = has_run and ((np.int64(buf[run_flags + c // 8]) >> (c % 8)) & 1) == 1
        if is_run:
            n_runs = _u16(buf, p)
            p += 2
            for r in range(n_runs):
                v0 = _u16(buf, p)
                ln = _u16(buf, p + 2)
                p += 4
                for v in range(v0, v0 + ln + 1):
                    bits[word0 + (v >> 6)] |= one << np.uint64(v & 63)
                n += ln + 1
        elif card <= ARRAY_MAX:
            for i in range(card):
                v = _u16(buf, p)
                bits[word0 + (v >> 6)] |= one << np.uint64(v & 63)
                p += 2
            n += card
        else:
            for w in range(1024):
                word = np.uint64(0)
                for b in range(8):
                    word |= np.uint64(buf[p + 8 * w + b]) << np.uint64(8 * b)
                bits[word0 + w] = word
            p += 8192
            n += card
    return n


@njit(cache=True, inline="always")
def _popcnt64(x: np.uint64) -> int:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def _near_bitset(a: np.ndarray, b: np.ndarray, n_words: int, window: int, exclude_self: bool) -> int:
    """|A & N(B)| der N(B) = posisjoner innenfor ±window fra B (uten B selv når exclude_self)."""
    hits = 0
    zero = np.uint64(0)
    for i in range(n_words):
        ai = a[i]
        if ai == zero:
            continue
        bi = b[i]
        prev = b[i - 1] if i > 0 else zero
        nxt = b[i + 1] if i + 1 < n_words else zero
        nbh = zero if exclude_self else bi
        for d in range(1, window + 1):
            sd = np.uint64(d)
            rd = np.uint64(64 - d)
            nbh |= (bi << sd) | (prev >> rd)   # q + d
            nbh |= (bi >> sd) | (nxt << rd)    # q - d
        hits += _popcnt64(ai & nbh)
    return hits


@njit(cache=True)
def near_books(
    arena_a: np.ndarray, starts_a: np.ndarray, ends_a: np.ndarray,
    arena_b: np.ndarray, starts_b: np.ndarray, ends_b: np.ndarray,
    window: int, exclude_self: bool, max_positions: int,
    dense_factor: float = 4.0,
) -> Tuple[int, int]:
    """
    NEAR over alle kandidatbøker i ett kall. starts/ends er byte-intervaller i arenaene.
    Per bok velges vei ut fra headerne alene:
      - fletting av sorterte posisjoner når ordene er spredte i boka
      - bitsett med shift/POPCNT når (|A| + |B|) * dense_factor > ord_i_bitsett * (2w + 1)
    """
    buf_a = np.empty(max_positions, dtype=np.uint32)
    buf_b = np.empty(max_positions, dtype=np.uint32)
    max_words = (max_positions >> 6) + 1024
    bits_a = np.zeros(max_words, dtype=np.uint64)
    bits_b = np.zeros(max_words, dtype=np.uint64)
    total = 0
    docs = 0
    same = exclude_self
    for i in range(starts_a.shape[0]):
        ca, span_a = roaring_card_span(arena_a, starts_a[i])
        if same:
            cb, span_b = ca, span_a
        else:
            cb, span_b = roaring_card_span(arena_b, starts_b[i])
        if ca < 0 or cb < 0:
            return -2, i
        span = max(span_a, span_b)
        dense = (ca + cb) * dense_factor > span * (2 * window + 1) and span <= max_words

        if dense:
            roaring_to_bitset(arena_a, starts_a[i], bits_a)
            if same:
                h = _near_bitset(bits_a, bits_a, span, window, True)
            else:
                roaring_to_bitset(arena_b, starts_b[i], bits_b)
                h = _near_bitset(bits_a, bits_b, span, window, False)
                bits_b[:span] = 0
            bits_a[:span] = 0
        else:
            na = roaring_decode(arena_a, starts_a[i], ends_a[i], buf_a)
            if na < 0:
                return na, i
            if same:
                h = _near_count(buf_a, na, buf_a, na, window, True)
            else:
                nb = roaring_decode(arena_b, starts_b[i], ends_b[i], buf_b)
                if nb < 0:
                    return nb, i
                h = _near_count(buf_a, na, buf_b, nb, window, False)
        if h > 0:
            total += h
            docs += 1
    return total, docs


def arena_offsets(offsets_blob: bytes | None, arena_len: int) -> np.ndarray:
    """offsets = NULL (df == 1) betyr at hele pos_arena er den ene boka."""
    if offsets_blob is None:
        return np.array([0, arena_len], dtype=np.uint32)
    return np.frombuffer(offsets_blob, dtype=np.uint32)


def book_slices(docs: BitMap, books: np.ndarray, offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Byte-intervallene i pos_arena for de gitte bøkene (sorterte, alle i docs)."""
    doc_arr = np.frombuffer(docs.to_array(), dtype=np.uint32)
    r = np.searchsorted(doc_arr, books)
    return offsets[r].astype(np.int64), offsets[r + 1].astype(np.int64)


def near_arena(
    docs_a: BitMap, offsets_a: np.ndarray, arena_a: bytes,
    docs_b: BitMap, offsets_b: np.ndarray, arena_b: bytes,
    *, window: int = 5, same_word: bool = False, max_positions: int = 1 << 22,
    dense_factor: float = 4.0,
) -> Tuple[int, int]:
    """
    NEAR_window mellom to ord gitt deres dokumentmengder, offsets og pos_arena.
    max_positions er største antall posisjoner for ett ord i én bok, og største
    posisjon i en bok for bitsett-veien. dense_factor styrer når bitsett velges
    (0 = aldri, stor verdi = alltid).
    """
    books = np.frombuffer((docs_a & docs_b).to_array(), dtype=np.uint32)
    if books.size == 0:
        return 0, 0
    sa, ea = book_slices(docs_a, books, offsets_a)
    sb, eb = book_slices(docs_b, books, offsets_b)
    a = np.frombuffer(arena_a, dtype=np.uint8)
    b = a if arena_b is arena_a else np.frombuffer(arena_b, dtype=np.uint8)
    if not 1 <= window <= 63:
        raise ValueError("window må være mellom 1 og 63")
    total, docs = near_books(a, sa, ea, b, sb, eb, window, same_word, max_positions, dense_factor)
    if total < 0:
        raise RuntimeError("Roaring-dekoding feilet (for liten buffer eller ukjent cookie)")
    return int(total), int(docs)
