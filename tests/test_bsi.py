"""
Tester for BSIPartialFunc (Bit-Sliced Roaring) mot vanlig PartialFunc.
Med heltallsverdier og scale=1.0 er BSI eksakt, så vi kan kreve likhet.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyroaring import BitMap

from bsi_roaring import BSIPartialFunc
from conftest import as_dict
from partial_func import PartialFunc

N_BITS = 10


def int_pf(rng: np.random.Generator, n: int) -> PartialFunc:
    ids = np.unique(
        np.concatenate([rng.choice(5_000, n // 2, replace=False),
                        rng.choice(np.arange(65_536, 200_000), n - n // 2, replace=False)])
    ).astype(np.uint32)
    vals = rng.integers(1, (1 << N_BITS) - 1, size=len(ids)).astype(np.float64)
    return PartialFunc.from_sorted_arrays(ids, vals)


@pytest.fixture
def ints(rng):
    f, g = int_pf(rng, 400), int_pf(rng, 300)
    return f, g, BSIPartialFunc.from_partial_func(f, n_bits=N_BITS, scale=1.0), \
        BSIPartialFunc.from_partial_func(g, n_bits=N_BITS, scale=1.0)


def test_roundtrip_exact(ints):
    f, _, bf, _ = ints
    assert as_dict(bf.to_partial_func()) == as_dict(f)
    assert len(bf) == len(f) and bf.n_bits == N_BITS


def test_getitem(ints):
    f, _, bf, _ = ints
    for w in list(f.dom)[:50]:
        assert bf[w] == f[w]
    assert bf[4_999_999] == 0.0


def test_sum_all_and_sum_on(ints):
    f, _, bf, _ = ints
    assert bf.sum_all() == float(np.sum(f.val))
    S = BitMap(list(f.dom)[::4] + [123_456_789])
    assert bf.sum_on(S) == float(np.sum(f.restrict(S).val))


def test_restrict(ints):
    f, _, bf, _ = ints
    S = BitMap(list(f.dom)[::3])
    assert as_dict(bf.restrict(S).to_partial_func()) == as_dict(f.restrict(S))
    assert len(bf.restrict(BitMap())) == 0


def test_add_matches_partial_func(ints):
    f, g, bf, bg = ints
    assert as_dict((bf + bg).to_partial_func()) == as_dict(f + g)


def test_add_requires_same_scale(ints):
    f, _, bf, _ = ints
    other = BSIPartialFunc.from_partial_func(f, n_bits=N_BITS, scale=2.0)
    with pytest.raises(ValueError):
        bf + other


def test_dot_matches_partial_func(ints):
    f, g, bf, bg = ints
    assert bf.dot(bg) == f.dot(g)


@pytest.mark.parametrize("threshold", [0.0, 1.0, 100.0, 512.0, 1000.0, 5000.0])
def test_threshold_bitmap(ints, threshold):
    f, _, bf, _ = ints
    expected = BitMap(w for w, v in as_dict(f).items() if v >= threshold)
    assert bf.threshold_bitmap(threshold) == expected


@pytest.mark.parametrize("threshold", [1023.0, 1500.0])
def test_threshold_bitmap_after_add_with_carry(ints, threshold):
    f, g, bf, bg = ints
    s = bf + bg
    expected = BitMap(w for w, v in as_dict(f + g).items() if v >= threshold)
    assert s.threshold_bitmap(threshold) == expected


def test_quantised_roundtrip_within_resolution(rng):
    """Med automatisk skala er feilen per verdi høyst et halvt kvantiseringstrinn."""
    f = PartialFunc.from_sorted_arrays(
        np.arange(0, 1_000, 3, dtype=np.uint32), rng.uniform(0.0, 7.0, size=334)
    )
    b = BSIPartialFunc.from_partial_func(f, n_bits=8)
    err = np.max(np.abs(b.to_partial_func().val - f.val))
    assert err <= 0.5 / b.scale + 1e-12


def test_empty():
    b = BSIPartialFunc.from_partial_func(PartialFunc.zero())
    assert len(b) == 0 and b.sum_all() == 0.0 and len(b.to_partial_func()) == 0
