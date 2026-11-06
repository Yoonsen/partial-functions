"""
Felles hjelpere for testene: tilfeldige partielle funksjoner og en
dict-basert referanseimplementasjon å sammenligne algebraen mot.

Alle data er syntetiske, slik at testene kan kjøres uten doc-to-vec,
metadata-roaring eller shard-filer.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from partial_func import PartialFunc

SEEDS = [0, 1, 2, 3, 4]


def random_pf(
    rng: np.random.Generator,
    n: int,
    *,
    universe: int = 200_000,
    positive: bool = True,
) -> PartialFunc:
    """
    Trekker n distinkte ID-er: halvparten i et tett område under 65 536
    (bitmap-containere i Roaring) og resten spredt opp til `universe`
    (array-containere), slik at begge containertypene blir testet.
    """
    n_dense = n // 2
    dense = rng.choice(5_000, size=n_dense, replace=False)
    sparse = rng.choice(np.arange(65_536, universe), size=n - n_dense, replace=False)
    ids = np.unique(np.concatenate([dense, sparse])).astype(np.uint32)
    vals = rng.uniform(0.1, 10.0, size=len(ids)) if positive else rng.normal(size=len(ids))
    return PartialFunc.from_sorted_arrays(ids, vals)


def as_dict(f: PartialFunc) -> Dict[int, float]:
    return {int(w): float(v) for w, v in zip(f.dom_array, f.val)}


def assert_pf_close(f: PartialFunc, ref: Dict[int, float], *, rtol: float = 1e-12) -> None:
    got = as_dict(f)
    assert set(got) == set(ref)
    for w, v in ref.items():
        assert got[w] == pytest.approx(v, rel=rtol, abs=1e-12)


@pytest.fixture(params=SEEDS)
def rng(request) -> np.random.Generator:
    return np.random.default_rng(request.param)


@pytest.fixture
def fg(rng):
    """To overlappende partielle funksjoner (delvis felles domene)."""
    f = random_pf(rng, 400)
    g = random_pf(rng, 300)
    return f, g


@pytest.fixture
def pq(fg):
    """To overlappende sannsynlighetsfordelinger."""
    f, g = fg
    return f.to_prob(), g.to_prob()
