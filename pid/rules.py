"""Data-driven extraction of robust antenna rules.

Pipeline
--------
1. Physics-aware features (log antenna ratio, log perimeter ratio, diode size,
   oxide thickness, layer, die radius).
2. Monotonic gradient boosting model of the Vt shift: the mean model is
   constrained to be non-decreasing in antenna ratio / perimeter / radius and
   non-increasing in diode size, which encodes known physics and makes the
   extracted rules consistent (no "allowed, forbidden, allowed again" artefacts).
3. Normalised split conformal calibration (Lei et al. 2018) on structures never
   seen in training, so the upper bound has a finite-sample coverage guarantee.
4. Rule extraction: for each configuration, the largest antenna ratio whose
   calibrated upper bound stays below the Vt spec at the worst-case die radius.
5. Comparison with the conventional "zero failure in the sweep" rule and with the
   ground truth known from the physical simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text

from .physics import OXIDES, antenna_current, damage, solve_oxide_stress
from .simulate import A_VT_MV_UM, DVT_SPEC_MV, MEAS_NOISE_MV
from .structures import COMB_FINGER_W_UM, LAYERS

BREAKDOWN_DVT_MV = 200.0   # censored value assigned to broken-down oxides
WORST_RADIUS_MM = 145.0
FEATURES = ["log_ar", "log_pr", "log_gate", "log_diode", "tox_nm", "layer_code", "radius_mm"]
MONOTONIC = [1, 1, 0, -1, 0, 0, 1]
CATEGORICAL = [False, False, False, False, False, True, False]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["log_ar"] = np.log10(df.antenna_ratio)
    x["log_pr"] = np.log10(df.perimeter_um / df.gate_um2)
    x["log_gate"] = np.log10(df.gate_um2)
    x["log_diode"] = np.log10(df.diode_um2 + 0.05)
    x["tox_nm"] = df.oxide.map(lambda o: OXIDES[o].tox_nm)
    x["layer_code"] = df.layer.map({l: i for i, l in enumerate(LAYERS)})
    x["radius_mm"] = df.radius_mm
    return x[FEATURES]


def target(df: pd.DataFrame) -> np.ndarray:
    return np.where(df.gate_leak_fail, BREAKDOWN_DVT_MV, df.dvt_mv.values)


def split_by_structure(df, test_size=0.25, calib_size=0.25, seed=0):
    """Train / calibration / test split by structure id (no die leakage)."""
    g = df.sid.values
    gss = GroupShuffleSplit(1, test_size=test_size, random_state=seed)
    trcal, te = next(gss.split(df, groups=g))
    gss2 = GroupShuffleSplit(1, test_size=calib_size / (1 - test_size), random_state=seed + 1)
    tr, cal = next(gss2.split(df.iloc[trcal], groups=g[trcal]))
    return trcal[tr], trcal[cal], te


Y_SCALE_MV = 5.0  # asinh transform scale: linear below ~5 mV, logarithmic above


def to_z(y):
    return np.arcsinh(np.asarray(y) / Y_SCALE_MV)


def from_z(z):
    return Y_SCALE_MV * np.sinh(z)


class RobustRuleModel:
    """Monotonic damage model + normalised split conformal upper bound.

    Note on design: scikit-learn's quantile loss re-estimates leaf values after
    each tree, which can break monotonic constraints. We therefore fit the
    monotonic model with squared error in a variance-stabilised space
    z = asinh(dVt / 5 mV), learn the local noise scale with a second model, and
    calibrate the normalised residual (Lei et al. 2018) on held-out structures.
    """

    def __init__(self, alpha: float = 0.95, seed: int = 0):
        self.alpha = alpha
        common = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                      min_samples_leaf=40, categorical_features=CATEGORICAL, random_state=seed)
        self.mean = HistGradientBoostingRegressor(monotonic_cst=MONOTONIC, **common)
        self.scale = HistGradientBoostingRegressor(**common)
        self.q_hat = 0.0

    def fit(self, df_train, df_cal):
        x, z = make_features(df_train), to_z(target(df_train))
        self.mean.fit(x, z)
        self.scale.fit(x, np.abs(z - self.mean.predict(x)))
        xc = make_features(df_cal)
        scores = (to_z(target(df_cal)) - self.mean.predict(xc)) / self._sigma(xc)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * self.alpha) / n)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self

    def _sigma(self, x):
        return np.maximum(self.scale.predict(x), 0.05)

    def median(self, df) -> np.ndarray:
        return from_z(self.mean.predict(make_features(df)))

    def upper(self, df) -> np.ndarray:
        x = make_features(df)
        return from_z(self.mean.predict(x) + self.q_hat * self._sigma(x))


def fail_classifier(df_train, seed=0):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, monotonic_cst=MONOTONIC,
        categorical_features=CATEGORICAL, random_state=seed)
    return clf.fit(make_features(df_train), df_train.fail.values)


class ProbabilisticRuleModel:
    """Monotonic failure classifier with isotonic recalibration.

    Isotonic regression is a non-decreasing map, so it keeps the monotonicity
    of the classifier while correcting its probability calibration.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, df_train, df_cal):
        self.clf = fail_classifier(df_train, self.seed)
        raw = self.clf.predict_proba(make_features(df_cal))[:, 1]
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(raw, df_cal.fail.values)
        return self

    def p_fail(self, df) -> np.ndarray:
        return self.iso.predict(self.clf.predict_proba(make_features(df))[:, 1])


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------
AR_GRID = np.geomspace(10, 100000, 241)


def config_grid(oxides=None, layers=None, diodes=(0.0, 0.2, 1.0, 4.0),
                shape="plate", gate_um2=1.0) -> pd.DataFrame:
    oxides = oxides or list(OXIDES)
    layers = layers or LAYERS
    return pd.DataFrame([dict(oxide=o, layer=l, diode_um2=d, shape=shape, gate_um2=gate_um2)
                         for o in oxides for l in layers for d in diodes])


def _expand(cfg: pd.DataFrame, radius_mm: float) -> pd.DataFrame:
    rows = cfg.loc[cfg.index.repeat(len(AR_GRID))].reset_index(drop=True)
    rows["cfg_id"] = np.repeat(np.arange(len(cfg)), len(AR_GRID))
    rows["antenna_ratio"] = np.tile(AR_GRID, len(cfg))
    a = rows.antenna_ratio * rows.gate_um2
    rows["perimeter_um"] = np.where(rows["shape"] == "plate", 4 * np.sqrt(a), 2 * a / COMB_FINGER_W_UM)
    rows["radius_mm"] = radius_mm
    return rows


def _max_allowed(rows: pd.DataFrame, ok: np.ndarray) -> np.ndarray:
    """Largest grid AR such that every smaller AR is also ok (monotone envelope)."""
    ok = ok.reshape(-1, len(AR_GRID))
    first_bad = np.where(ok.all(1), len(AR_GRID), np.argmin(ok, axis=1))
    return np.where(first_bad == 0, np.nan, AR_GRID[np.maximum(first_bad - 1, 0)])


def ml_rules(model: RobustRuleModel, cfg: pd.DataFrame, spec=DVT_SPEC_MV,
             radius_mm=WORST_RADIUS_MM) -> np.ndarray:
    rows = _expand(cfg, radius_mm)
    return _max_allowed(rows, model.upper(rows) <= spec)


def prob_rules(model: ProbabilisticRuleModel, cfg: pd.DataFrame, p_fail_max=0.05,
               radius_mm=WORST_RADIUS_MM) -> np.ndarray:
    rows = _expand(cfg, radius_mm)
    return _max_allowed(rows, model.p_fail(rows) <= p_fail_max)


def true_rules(cfg: pd.DataFrame, p_fail_max=0.05, n_mc=400, seed=7,
               radius_mm=WORST_RADIUS_MM) -> np.ndarray:
    """Ground truth from the simulator: largest AR with P(fail) <= p_fail_max."""
    rng = np.random.default_rng(seed)
    rows = _expand(cfg, radius_mm)
    p_fail = np.zeros(len(rows))
    for ox_name, idx in rows.groupby("oxide").indices.items():
        ox = OXIDES[ox_name]
        sub = rows.iloc[idx]
        fails = np.zeros(len(sub))
        for _ in range(n_mc):
            wf = rng.lognormal(0.0, 0.15)
            i_ant = antenna_current(sub.antenna_ratio * sub.gate_um2, sub.perimeter_um,
                                    sub.layer.values, sub.radius_mm, wf)
            _, j = solve_oxide_stress(i_ant, sub.gate_um2.values, sub.diode_um2.values, ox, 50)
            _, d, b = damage(j, sub.gate_um2.values, ox, rng)
            sigma = np.sqrt(2) * A_VT_MV_UM / np.sqrt(sub.gate_um2.values)
            d = d + rng.normal(0, sigma) + rng.normal(0, MEAS_NOISE_MV, len(sub))
            fails += (np.abs(d) > DVT_SPEC_MV) | b
        p_fail[idx] = fails / n_mc
    return _max_allowed(rows, p_fail <= p_fail_max)


def conventional_rules(data: pd.DataFrame, cfg: pd.DataFrame) -> np.ndarray:
    """Classical practice: on the AR sweep of the reference structure, the rule is
    the largest AR below the first AR at which any failure is observed."""
    out = []
    sweep = data[(data["shape"] == "plate") & (data.gate_um2 == 1.0)]
    for c in cfg.itertuples():
        s = sweep[(sweep.oxide == c.oxide) & (sweep.layer == c.layer) & (sweep.diode_um2 == c.diode_um2)]
        per_ar = s.groupby("antenna_ratio").fail.any().sort_index()
        if per_ar.empty:
            out.append(np.nan)
            continue
        bad = per_ar[per_ar].index
        if len(bad) == 0:
            out.append(per_ar.index.max())
        else:
            ok = per_ar.index[per_ar.index < bad.min()]
            out.append(ok.max() if len(ok) else np.nan)
    return np.array(out, dtype=float)


def explainable_rules(df: pd.DataFrame, max_depth: int = 4) -> str:
    """Shallow decision tree on interpretable features, exported as text rules."""
    x = pd.DataFrame({
        "log10_AR": np.log10(df.antenna_ratio),
        "diode_um2": df.diode_um2,
        "tox_nm": df.oxide.map(lambda o: OXIDES[o].tox_nm),
        "radius_mm": df.radius_mm,
    })
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=200,
                                  class_weight="balanced", random_state=0)
    tree.fit(x, df.fail)
    return export_text(tree, feature_names=list(x.columns), decimals=2)
