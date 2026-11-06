"""
Tester for PartialFunc: konstruksjon, vektorromsaksiomer, snitt-operasjoner,
distanser og ettersporbar dekomponering, alt mot en dict-basert referanse.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pyroaring import BitMap

from conftest import as_dict, assert_pf_close, random_pf
from partial_func import PartialFunc, SparseWorkspace, _bitblock_intersect_dot_and_jsd


# ---------------------------------------------------------------------------
# Referanser over unionen (det algebraen skal slippe å regne ut)
# ---------------------------------------------------------------------------

def ref_jsd(p: dict, q: dict) -> float:
    total = 0.0
    for w in set(p) | set(q):
        a, b = p.get(w, 0.0), q.get(w, 0.0)
        m = 0.5 * (a + b)
        if a > 0:
            total += 0.5 * a * math.log2(a / m)
        if b > 0:
            total += 0.5 * b * math.log2(b / m)
    return total


def ref_tv(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p.get(w, 0.0) - q.get(w, 0.0)) for w in set(p) | set(q))


def ref_hellinger(p: dict, q: dict) -> float:
    bc = sum(math.sqrt(p[w] * q[w]) for w in set(p) & set(q))
    return math.sqrt(max(0.0, 1.0 - bc))


# ---------------------------------------------------------------------------
# Konstruksjon og oppslag
# ---------------------------------------------------------------------------

def test_zero_is_empty():
    z = PartialFunc.zero()
    assert len(z) == 0 and len(z.dom) == 0 and z.val.size == 0


def test_from_dict_sorts_and_evaluates():
    f = PartialFunc.from_dict({70_000: 2.0, 3: 1.5, 10: -1.0})
    assert list(f.dom_array) == [3, 10, 70_000]
    assert f[3] == 1.5 and f[10] == -1.0 and f[70_000] == 2.0
    assert f[4] == 0.0
    assert 3 in f and 4 not in f


def test_from_dict_empty_is_zero():
    assert len(PartialFunc.from_dict({})) == 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        PartialFunc(BitMap([1, 2, 3]), np.array([1.0, 2.0]))


def test_dom_array_is_numpy_uint32(fg):
    f, _ = fg
    arr = f.dom_array
    assert isinstance(arr, np.ndarray) and arr.dtype == np.uint32
    assert np.array_equal(arr, np.array(list(f.dom), dtype=np.uint32))


def test_prune_zeros():
    f = PartialFunc.from_dict({1: 0.0, 2: 3.0, 5: 0.0})
    g = f.prune_zeros()
    assert as_dict(g) == {2: 3.0}
    assert len(f) == 3, "prune_zeros skal ikke endre originalen"


# ---------------------------------------------------------------------------
# Vektorromsaksiomer (addisjon over unionen)
# ---------------------------------------------------------------------------

def test_add_matches_reference(fg):
    f, g = fg
    a, b = as_dict(f), as_dict(g)
    ref = {w: a.get(w, 0.0) + b.get(w, 0.0) for w in set(a) | set(b)}
    assert_pf_close(f + g, ref)


def test_linear_comb_matches_reference(fg):
    f, g = fg
    a, b = as_dict(f), as_dict(g)
    ref = {w: 2.5 * a.get(w, 0.0) - 0.75 * b.get(w, 0.0) for w in set(a) | set(b)}
    assert_pf_close(f.linear_comb(g, 2.5, -0.75), ref)


def test_add_commutative(fg):
    f, g = fg
    assert_pf_close(f + g, as_dict(g + f))


def test_add_associative(rng):
    f, g, h = (random_pf(rng, 200) for _ in range(3))
    assert_pf_close((f + g) + h, as_dict(f + (g + h)), rtol=1e-9)


def test_zero_is_additive_identity(fg):
    f, _ = fg
    assert_pf_close(f + PartialFunc.zero(), as_dict(f))
    assert_pf_close(PartialFunc.zero() + f, as_dict(f))


def test_additive_inverse_has_explicit_zeros(fg):
    f, _ = fg
    d = f - f
    assert d.dom == f.dom, "f - f beholder domenet (eksplisitte nuller, se AGENTS.md)"
    assert np.all(d.val == 0.0)
    assert len(d.prune_zeros()) == 0


def test_scalar_distributivity(fg):
    f, g = fg
    assert_pf_close(3.0 * (f + g), as_dict(3.0 * f + 3.0 * g), rtol=1e-9)
    assert_pf_close((2.0 + 5.0) * f, as_dict(2.0 * f + 5.0 * f), rtol=1e-9)


def test_neg_and_sub(fg):
    f, g = fg
    assert_pf_close(f - g, as_dict(f + (-g)))


# ---------------------------------------------------------------------------
# Snitt-operasjoner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["gallop", "twopointer", "rank"])
def test_align_intersection_methods_agree(fg, method):
    f, g = fg
    a, b = as_dict(f), as_dict(g)
    inter, vf, vg = f.align_intersection(g, method=method)
    common = sorted(set(a) & set(b))
    assert list(inter) == common
    assert np.allclose(vf, [a[w] for w in common])
    assert np.allclose(vg, [b[w] for w in common])


def test_align_intersection_unknown_method(fg):
    f, g = fg
    with pytest.raises(ValueError):
        f.align_intersection(g, method="nope")


def test_pointwise_product_on_intersection(fg):
    f, g = fg
    a, b = as_dict(f), as_dict(g)
    assert_pf_close(f * g, {w: a[w] * b[w] for w in set(a) & set(b)})


def test_pointwise_quotient_on_intersection(fg):
    f, g = fg
    a, b = as_dict(f), as_dict(g)
    assert_pf_close(f / g, {w: a[w] / b[w] for w in set(a) & set(b)})


def test_quotient_by_zero_gives_inf():
    """Dokumentert oppførsel (AGENTS.md 2A): f / g sjekker ikke for g(w) = 0."""
    f = PartialFunc.from_dict({1: 1.0, 2: 2.0})
    g = PartialFunc.from_dict({1: 0.0, 2: 4.0})
    with np.errstate(divide="ignore"):
        q = f / g
    assert math.isinf(q[1]) and q[2] == 0.5


def test_restrict(fg):
    f, _ = fg
    a = as_dict(f)
    S = BitMap(list(a)[::3] + [999_999])
    assert_pf_close(f.restrict(S), {w: a[w] for w in a if w in S})
    assert len(f.restrict(BitMap())) == 0


def test_dot(fg):
    f, g = fg
    a, b = as_dict(f), as_dict(g)
    ref = sum(a[w] * b[w] for w in set(a) & set(b))
    assert f.dot(g) == pytest.approx(ref)
    assert (f @ g) == pytest.approx(ref)


def test_dot_disjoint_is_zero():
    f = PartialFunc.from_dict({1: 1.0})
    g = PartialFunc.from_dict({2: 1.0})
    assert f.dot(g) == 0.0


# ---------------------------------------------------------------------------
# Fordelinger og distanser over snittet (mot referanse over unionen)
# ---------------------------------------------------------------------------

def test_to_prob_sums_to_one(fg):
    f, _ = fg
    assert float(np.sum(f.to_prob().val)) == pytest.approx(1.0)


def test_to_prob_nonpositive_is_zero():
    assert len(PartialFunc.from_dict({1: 0.0}).to_prob()) == 0
    assert len(PartialFunc.zero().to_prob()) == 0


def test_jsd_matches_union_reference(pq):
    p, q = pq
    assert p.jsd(q) == pytest.approx(ref_jsd(as_dict(p), as_dict(q)), abs=1e-12)


def test_tv_matches_union_reference(pq):
    p, q = pq
    assert p.total_variation(q) == pytest.approx(ref_tv(as_dict(p), as_dict(q)), abs=1e-12)


def test_hellinger_matches_reference(pq):
    p, q = pq
    assert p.hellinger(q) == pytest.approx(ref_hellinger(as_dict(p), as_dict(q)), abs=1e-12)


@pytest.mark.parametrize("name", ["jsd", "total_variation", "hellinger"])
def test_distance_symmetric_and_bounded(pq, name):
    p, q = pq
    d_pq, d_qp = getattr(p, name)(q), getattr(q, name)(p)
    assert d_pq == pytest.approx(d_qp, abs=1e-12)
    assert 0.0 <= d_pq <= 1.0


@pytest.mark.parametrize("name", ["jsd", "total_variation", "hellinger"])
def test_distance_identity_and_disjoint(name):
    p = PartialFunc.from_dict({1: 0.25, 2: 0.75})
    q = PartialFunc.from_dict({3: 0.5, 4: 0.5})
    assert getattr(p, name)(p) == pytest.approx(0.0, abs=1e-7)
    assert getattr(p, name)(q) == 1.0


def test_pmi_matches_reference(rng):
    f = random_pf(rng, 300)
    bg = random_pf(rng, 2_000).to_prob()
    pf_, b = as_dict(f.to_prob()), as_dict(bg)
    common = set(pf_) & set(b)

    ref = {w: math.log2(pf_[w] / b[w] ** 0.75) - math.log2(5.0) for w in common}
    assert_pf_close(f.pmi(bg, alpha=0.75, shift_k=5.0, positive=False), ref, rtol=1e-9)

    ref_pos = {w: v for w, v in ref.items() if v > 0.0}
    assert_pf_close(f.pmi(bg, alpha=0.75, shift_k=5.0, positive=True), ref_pos, rtol=1e-9)


# ---------------------------------------------------------------------------
# Ettersporbarhet: summen av forklaringen er lik skalaren
# ---------------------------------------------------------------------------

def test_explain_dot_sums_to_dot(fg):
    f, g = fg
    assert float(np.sum(f.explain_dot(g).val)) == pytest.approx(f.dot(g))


def test_explain_overlap_sums_to_one_minus_tv(pq):
    p, q = pq
    assert float(np.sum(p.explain_overlap(q).val)) == pytest.approx(1.0 - p.total_variation(q))


def test_explain_jsd_sums_to_jsd_over_union(pq):
    p, q = pq
    e = p.explain_jsd_divergence(q)
    assert e.dom == (p.dom | q.dom)
    assert float(np.sum(e.val)) == pytest.approx(p.jsd(q), abs=1e-12)


def test_explain_jsd_disjoint():
    p = PartialFunc.from_dict({1: 1.0})
    q = PartialFunc.from_dict({2: 1.0})
    assert float(np.sum(p.explain_jsd_divergence(q).val)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("largest", [True, False])
def test_top_k(fg, largest):
    f, _ = fg
    ref = sorted(as_dict(f).items(), key=lambda kv: kv[1], reverse=largest)[:7]
    got = f.top_k(7, largest=largest)
    assert [w for w, _ in got] == [w for w, _ in ref]


def test_top_k_edge_cases():
    f = PartialFunc.from_dict({1: 1.0, 2: 2.0})
    assert PartialFunc.zero().top_k(3) == []
    assert f.top_k(0) == []
    assert [w for w, _ in f.top_k(10)] == [2, 1]


# ---------------------------------------------------------------------------
# SparseWorkspace og Numba-kjernene skal gi samme svar som PartialFunc
# ---------------------------------------------------------------------------

def test_workspace_dot_and_distances(pq):
    p, q = pq
    ws = SparseWorkspace(universe_size=200_000)
    ws.scatter(p)
    assert ws.dot_with_active(q) == pytest.approx(p.dot(q))

    # Merk: workspace-kjernen returnerer ferdig JSD og TV, mens bitblock-kjernen
    # under returnerer de rå snitt-summene.
    dot, jsd, tv, n_common = ws.dot_and_jsd_with_active(q)
    assert dot == pytest.approx(p.dot(q))
    assert n_common == len(p.dom & q.dom)
    assert jsd == pytest.approx(p.jsd(q), abs=1e-12)
    assert tv == pytest.approx(p.total_variation(q), abs=1e-12)


def test_workspace_scatter_clears_previous(fg):
    f, g = fg
    ws = SparseWorkspace(universe_size=200_000)
    ws.scatter(f)
    ws.scatter(g)
    assert ws.dot_with_active(f) == pytest.approx(g.dot(f))
    ws.clear()
    assert not np.any(ws.work)


def test_workspace_gather_on_bitmap(fg):
    f, _ = fg
    ws = SparseWorkspace(universe_size=200_000)
    ws.scatter(f)
    S = BitMap(list(f.dom)[:10] + [5_001])
    expected = [f[w] for w in S]
    assert np.allclose(ws.gather_on_bitmap(S), expected)


def test_bitblock_kernel_matches(pq):
    p, q = pq
    dot, jsd_sum, tv_sum, n_common = _bitblock_intersect_dot_and_jsd(
        *p.bitblocks, p.val, *q.bitblocks, q.val
    )
    assert dot == pytest.approx(p.dot(q))
    assert n_common == len(p.dom & q.dom)
    assert 1.0 + 0.5 * jsd_sum == pytest.approx(p.jsd(q), abs=1e-12)
    assert 1.0 - tv_sum == pytest.approx(p.total_variation(q), abs=1e-12)
