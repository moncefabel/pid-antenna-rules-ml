"""Step 3: learn the damage response, extract antenna rules, evaluate them.

Three rule extraction strategies are compared against the simulator ground truth:
  * conventional : largest AR before the first observed failure on the AR sweep
  * ml_prob      : calibrated monotonic failure classifier, P(fail) <= 5 % at wafer edge
  * ml_guaranteed: conformal upper bound of dVt <= spec at wafer edge (coverage guarantee)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

from pid.rules import (FEATURES, ProbabilisticRuleModel, RobustRuleModel, config_grid,
                       conventional_rules, explainable_rules, make_features, ml_rules,
                       prob_rules, split_by_structure, target, true_rules)
from pid.simulate import DVT_SPEC_MV

out = Path("results")
data = pd.read_csv(out / "measurements.csv.gz")
tr, cal, te = split_by_structure(data)
d_tr, d_cal, d_te = data.iloc[tr], data.iloc[cal], data.iloc[te]
print(f"train {d_tr.sid.nunique()} / calib {d_cal.sid.nunique()} / test {d_te.sid.nunique()} structures")

# ---------------------------------------------------------------- models
guar = RobustRuleModel(alpha=0.95).fit(d_tr, d_cal)
coverage = float(np.mean(target(d_te) <= guar.upper(d_te)))
prob = ProbabilisticRuleModel().fit(d_tr, d_cal)
p_te = prob.p_fail(d_te)
auc = float(roc_auc_score(d_te.fail, p_te))
brier = float(brier_score_loss(d_te.fail, p_te))
print(f"conformal dVt upper bound: test coverage {coverage:.3f} (target 0.95)")
print(f"failure model: test AUC {auc:.3f}, Brier {brier:.4f}")

xs = make_features(d_te).sample(8000, random_state=0)
imp = permutation_importance(prob.clf, xs, d_te.loc[xs.index].fail, scoring="roc_auc",
                             n_repeats=5, random_state=0)
importance = {k: float(v) for k, v in sorted(zip(FEATURES, imp.importances_mean.round(4)),
                                             key=lambda kv: -kv[1])}

# ---------------------------------------------------------------- rules on the reference structure
cfg = config_grid()
cfg["rule_truth"] = true_rules(cfg)
cfg["rule_conventional"] = conventional_rules(data, cfg)
cfg["rule_ml_prob"] = prob_rules(prob, cfg)
cfg["rule_ml_guaranteed"] = ml_rules(guar, cfg)
cfg.to_csv(out / "antenna_rules.csv", index=False)


def score(rule, truth):
    r = pd.Series(rule / truth).dropna()
    return {"unsafe_rate": round(float(np.mean(r > 1.15)), 3),
            "median_ratio_to_truth": round(float(np.median(r)), 3)}


summary = {"reference_structure_full_data": {
    m: score(cfg[f"rule_{m}"], cfg.rule_truth) for m in ["conventional", "ml_prob", "ml_guaranteed"]}}

# ---------------------------------------------------------------- data budget experiment
budget = []
rng = np.random.default_rng(0)
for n_dies in [3, 6, 12, 25]:
    for rep in range(5):
        wafer = rng.choice(data.wafer.unique())
        dies = rng.choice(data.die.unique(), n_dies, replace=False)
        sub = data[(data.wafer == wafer) & data.die.isin(dies)]
        stri, scal, _ = split_by_structure(sub, test_size=0.01, calib_size=0.25, seed=rep)
        pm = ProbabilisticRuleModel(seed=rep).fit(sub.iloc[stri], sub.iloc[scal])
        for m, rule in [("conventional", conventional_rules(sub, cfg)), ("ml_prob", prob_rules(pm, cfg))]:
            s = score(rule, cfg.rule_truth.values)
            budget.append(dict(n_dies=n_dies, rep=rep, method=m, **s))
budget = pd.DataFrame(budget)
budget_summary = budget.groupby(["n_dies", "method"])[["unsafe_rate", "median_ratio_to_truth"]].mean().round(3)
print(budget_summary)
summary["data_budget"] = budget_summary.reset_index().to_dict(orient="records")

# ---------------------------------------------------------------- generalisation to un-swept structures
# The AR sweep only uses 1 um2 plates. We test the models on structures the
# conventional method never characterised: comb antennas and a 4 um2 gate.
gen = {}
for name, shape, gate in [("comb_1um2_gate", "comb", 1.0), ("plate_4um2_gate", "plate", 4.0),
                          ("comb_0p2um2_gate", "comb", 0.2)]:
    cu = config_grid(shape=shape, gate_um2=gate)
    cu["rule_truth"] = true_rules(cu)
    cu["rule_ml_prob"] = prob_rules(prob, cu)
    cu["rule_ml_guaranteed"] = ml_rules(guar, cu)
    cu.to_csv(out / f"antenna_rules_{name}.csv", index=False)
    gen[name] = {m: score(cu[f"rule_{m}"], cu.rule_truth) for m in ["ml_prob", "ml_guaranteed"]}
    gen[name]["truth_defined_share"] = round(float(cu.rule_truth.notna().mean()), 3)
summary["unswept_structures"] = gen

(out / "explainable_rules.txt").write_text(explainable_rules(d_tr))
metrics = {"dvt_upper_bound_test_coverage": coverage, "fail_auc": auc, "fail_brier": brier,
           "permutation_importance": importance, "rules": summary}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(summary["reference_structure_full_data"], indent=2))
print(json.dumps(summary["unswept_structures"], indent=2))

# ---------------------------------------------------------------- figures
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
jit = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(6, 4))
s = data[(data.oxide == "thick") & (data.layer == "M2") & (data["shape"] == "plate") & (data.gate_um2 == 1.0)]
for d, g in s.groupby("diode_um2"):
    ax.scatter(g.antenna_ratio * np.exp(jit.normal(0, .03, len(g))), np.clip(g.dvt_mv, -5, 250),
               s=4, alpha=.35, label=f"diode {d:g} um2")
ax.axhline(DVT_SPEC_MV, color="k", ls="--", lw=1, label="spec 20 mV")
ax.set(xscale="log", xlabel="Antenna ratio (antenna area / gate area)", ylabel="dVt (mV)",
       title="Thick oxide, M2 plate: Vt shift vs antenna ratio")
ax.legend(markerscale=3, fontsize=7)
fig.tight_layout(); fig.savefig(out / "fig1_dvt_vs_ar.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.5))
s2 = data[(data.antenna_ratio > 1000) & (data.antenna_ratio < 5000) & (data.diode_um2 == 0)]
s2.groupby(pd.cut(s2.radius_mm, np.linspace(0, 150, 7)), observed=True).fail.mean().plot.bar(ax=ax, color="tab:red")
ax.set(xlabel="Die radius (mm)", ylabel="Fail rate", title="Edge effect of plasma non-uniformity (AR 1k to 5k, no diode)")
fig.tight_layout(); fig.savefig(out / "fig2_radial_effect.png"); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
cols = [("rule_truth", "ground truth", "0.3"), ("rule_conventional", "conventional sweep", "tab:orange"),
        ("rule_ml_prob", "ML probabilistic", "tab:blue"), ("rule_ml_guaranteed", "ML guaranteed (conformal)", "tab:green")]
for ax, ox in zip(axes, ["thin", "medium", "thick"]):
    c = cfg[(cfg.layer == "M2") & (cfg.oxide == ox)]
    x = np.arange(len(c))
    for k, (col, lab, colr) in enumerate(cols):
        ax.bar(x + (k - 1.5) * .2, c[col], .2, label=lab, color=colr)
    ax.set_xticks(x, [f"{d:g}" for d in c.diode_um2]); ax.set_yscale("log")
    ax.set(title=f"{ox} oxide, M2", xlabel="diode area (um2)")
axes[0].set_ylabel("max antenna ratio"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(out / "fig3_rules_comparison.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 3.6))
for m, colr in [("conventional", "tab:orange"), ("ml_prob", "tab:blue")]:
    b = budget[budget.method == m].groupby("n_dies").unsafe_rate.agg(["mean", "std"])
    ax.errorbar(b.index, b["mean"], yerr=b["std"], marker="o", capsize=3, label=m, color=colr)
ax.set(xlabel="dies measured (single wafer)", ylabel="share of unsafe rules",
       title="Rule safety vs test budget")
ax.legend(); fig.tight_layout(); fig.savefig(out / "fig4_data_budget.png"); plt.close(fig)
print("figures written to results/")
