import numpy as np
import pandas as pd

from pid.physics import OXIDES, antenna_current, damage, solve_oxide_stress
from pid.rules import _expand, _max_allowed, config_grid
from pid.structures import TestStructure


def _dvt(ar, diode=0.0, oxide="medium", r=100.0):
    ar = np.atleast_1d(np.asarray(ar, dtype=float))
    i = antenna_current(ar, 4 * np.sqrt(ar), np.array(["M1"] * len(ar)), np.full(len(ar), r))
    _, j = solve_oxide_stress(i, np.ones(len(ar)), np.full(len(ar), diode), OXIDES[oxide])
    return damage(j, np.ones(len(ar)), OXIDES[oxide])[1]


def test_damage_increases_with_antenna_ratio():
    d = _dvt(np.geomspace(10, 1e5, 30))
    assert np.all(np.diff(d) >= -1e-9)


def test_diode_reduces_damage():
    ar = np.geomspace(100, 1e5, 20)
    assert np.all(_dvt(ar, diode=1.0) <= _dvt(ar, diode=0.0) + 1e-9)


def test_edge_dies_are_worse():
    assert _dvt(3000, r=145)[0] > _dvt(3000, r=0)[0]


def test_current_balance_is_satisfied():
    ox = OXIDES["thick"]
    i = np.array([1e-9, 1e-8])
    v, j = solve_oxide_stress(i, np.ones(2), np.zeros(2), ox)
    i_back = 1e-8 * (j + ox.g_leak * v)
    assert np.allclose(i_back, i, rtol=1e-4)


def test_comb_has_more_perimeter_than_plate():
    p = TestStructure("a", "thin", "M1", "plate", 1.0, 1000, 0.0)
    c = TestStructure("b", "thin", "M1", "comb", 1.0, 1000, 0.0)
    assert c.perimeter_um > 5 * p.perimeter_um


def test_max_allowed_uses_first_violation():
    cfg = config_grid(oxides=["thin"], layers=["M1"], diodes=(0.0,))
    rows = _expand(cfg, 100.0)
    ok = rows.antenna_ratio.values < 1000
    ok[5] = False  # an isolated violation must cap the rule
    assert _max_allowed(rows, ok)[0] < 1000 and _max_allowed(rows, ok)[0] == rows.antenna_ratio.values[4]
