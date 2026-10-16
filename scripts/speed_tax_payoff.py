#!/usr/bin/env python
"""Speed Tax step 3: does Combine-relative speed pay off on Sundays?

    python scripts/speed_tax_payoff.py

Uses data/processed/route_usage.parquet (from scripts/speed_tax.py). Writes docs/speed_tax_payoff.md
and data/processed/speed_tax_players_v2.parquet.

1. Translation: log(game peak speed / burst) ~ beta * log(Combine capacity) + context + (1 | player).
   beta is the Combine-to-game elasticity; the player intercept is the denominator-corrected tax.
2. Play-level payoff: within-player (player fixed effects) separation and target on speed, by route
   family; then whether separation is explained better by Combine-relative speed than absolute speed.
3. Player level: do Combine capacities or game taxes predict separation-over-expected and target
   rate beyond draft slot + forty + size? Nested CV with bootstrap CIs.
"""
from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from speed_tax import ROUTES, eligible, md, prepare  # noqa: E402

PROC = ROOT / "data/processed"
DOC = ROOT / "docs/speed_tax_payoff.md"
FAMILY = {"GO": "vertical", "POST": "vertical", "CORNER": "vertical", "WHEEL": "vertical",
          "IN": "breaking", "OUT": "breaking", "CROSS": "breaking",
          "HITCH": "short", "SLANT": "short", "FLAT": "short", "ANGLE": "short"}
CONTEXT = "C(routeRan) + bs(tWindow, df=4) + C(align) + C(manZone) + cushionF + cushionMissing + room + motion + C(down) + yardsToGo + C(season)"
RNG = np.random.default_rng(2027)


def mixed(formula, d, label):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(formula, d, groups=d["nflId"]).fit(method=["bfgs", "powell"], reml=True)
    re = pd.Series({k: v.iloc[0] for k, v in m.random_effects.items()}, name=label)
    return m, re


def coef_row(m, term, label, n_players=None):
    ci = m.conf_int().loc[term]
    return {"model": label, "term": term, "coef": m.params[term], "lo95": ci[0], "hi95": ci[1],
            "p": m.pvalues[term], "n": int(m.nobs), "players": n_players}


# ------------------------------------------------------------------ 1. translation
def translation(d):
    rows, taxes = [], []
    sp = eligible(d[(d.windowEnd == "pass") & d.peakSpeed.gt(0)], "speedFrac", 40).copy()
    sp["logGame"], sp["logCap"] = np.log(sp.peakSpeed), np.log(sp.vmax)
    sp["logCapC"] = sp.logCap - sp.logCap.mean()
    m, re = mixed(f"logGame ~ logCapC + {CONTEXT}", sp, "speedTaxLog")
    rows.append(coef_row(m, "logCapC", "log game peak speed ~ log Combine vmax", sp.nflId.nunique()))
    taxes.append(re)
    bu = d[(d.windowEnd == "pass") & (d.sAtSnap < 1.0) & (d.motion == 0) & d.burst.gt(0)]
    bu = eligible(bu, "burstFrac", 40).copy()
    bu["logGame"], bu["logCap"] = np.log(bu.burst), np.log(bu.burst2_10)
    bu["logCapC"] = bu.logCap - bu.logCap.mean()
    ctx_b = CONTEXT.replace("bs(tWindow, df=4) + ", "").replace("motion + ", "")
    m, re = mixed(f"logGame ~ logCapC + {ctx_b}", bu, "burstTaxLog")
    rows.append(coef_row(m, "logCapC", "log game burst (2-10 yd) ~ log Combine burst2_10", bu.nflId.nunique()))
    taxes.append(re)
    # attenuation check: Combine burst has retest ICC ~0.82, so the true elasticity is about beta / 0.82
    return pd.DataFrame(rows), pd.concat(taxes, axis=1)


# ------------------------------------------------------------------ 2. play level
def play_level(d):
    s = d[(d.windowEnd == "pass") & d.separationAtPassForward.notna() & d.routeRan.isin(FAMILY)
          & d.speedFrac.notna()].copy()
    s["family"] = s.routeRan.map(FAMILY)
    # within-player standardized speed: z of peak speed within player x route (removes player & route means)
    g = s.groupby(["nflId", "routeRan"]).peakSpeed
    s["speedZ"] = (s.peakSpeed - g.transform("mean")) / g.transform("std")
    s = s[s.groupby(["nflId", "routeRan"]).peakSpeed.transform("count") >= 5].dropna(subset=["speedZ"])
    rows = []
    for outcome, lab in (("separationAtPassForward", "separation (yd)"), ("targetI", "target (LPM)")):
        s["targetI"] = s.target.astype(int)
        for fam in ("vertical", "breaking", "short"):
            x = s[s.family == fam]
            m = smf.ols(f"{outcome} ~ speedZ + {CONTEXT} + C(nflId)", x).fit(
                cov_type="cluster", cov_kwds={"groups": x["nflId"]})
            rows.append({"outcome": lab, "family": fam, **{k: v for k, v in coef_row(m, "speedZ", "").items()
                                                          if k in ("coef", "lo95", "hi95", "p", "n")},
                         "players": x.nflId.nunique()})
    within = pd.DataFrame(rows)

    # absolute vs Combine-relative speed (between + within, mixed model, ML for comparable likelihoods)
    comp = []
    for fam in ("vertical", "breaking", "short"):
        x = s[s.family == fam].copy()
        x["peakC"] = x.peakSpeed - x.peakSpeed.mean()
        x["fracC"] = (x.speedFrac - x.speedFrac.mean()) * 10  # per 0.1 of capacity
        x["vmaxC"] = x.vmax - x.vmax.mean()
        fits = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, rhs in (("absolute speed", "peakC"), ("Combine-relative speed", "fracC"),
                              ("absolute + Combine vmax", "peakC + vmaxC")):
                fits[name] = smf.mixedlm(f"separationAtPassForward ~ {rhs} + {CONTEXT}", x,
                                         groups=x["nflId"]).fit(reml=False, method=["bfgs", "powell"])
        base = fits["absolute speed"].llf
        for name, m in fits.items():
            term = {"absolute speed": "peakC", "Combine-relative speed": "fracC", "absolute + Combine vmax": "vmaxC"}[name]
            comp.append({"family": fam, "model": name, "logLik - absolute": m.llf - base,
                         "key term": term, "coef": m.params[term], "p": m.pvalues[term], "n": int(m.nobs)})
    return within, pd.DataFrame(comp), s


# ------------------------------------------------------------------ 3. player level
def player_outcomes(d):
    """Per-player separation over expected (context-only model residual) and target rate over expected."""
    s = d[(d.windowEnd == "pass") & d.separationAtPassForward.notna() & (d.routeRan != "SCREEN")].copy()
    s["targetI"] = s.target.astype(int)
    out = {}
    for y, name in (("separationAtPassForward", "sepOE"), ("targetI", "targetOE")):
        m = smf.ols(f"{y} ~ {CONTEXT}", s).fit()
        s[name] = s[y] - m.fittedvalues
        g = s.groupby("nflId")[name]
        n, mean, var_w = g.size(), g.mean(), g.var().mean()
        var_b = max(mean.var() - (var_w / n).mean(), 1e-9)
        out[name] = mean * var_b / (var_b + var_w / n)  # empirical-Bayes shrinkage
    out["routes"] = s.groupby("nflId").size()
    return pd.DataFrame(out)


def nested_cv(df, y, base, add, reps=200, folds=5):
    """CV R^2 of base vs base+add (repeated K-fold), with a bootstrap CI on the difference."""
    x = df.dropna(subset=[y] + base + add)
    def cvr2(cols, seed):
        kf = KFold(folds, shuffle=True, random_state=seed)
        pred = np.zeros(len(x))
        for tr, te in kf.split(x):
            pred[te] = LinearRegression().fit(x.iloc[tr][cols], x.iloc[tr][y]).predict(x.iloc[te][cols])
        return 1 - ((x[y] - pred) ** 2).sum() / ((x[y] - x[y].mean()) ** 2).sum(), pred
    r_b = [cvr2(base, s)[0] for s in range(20)]
    r_a = [cvr2(base + add, s)[0] for s in range(20)]
    # bootstrap the delta using out-of-fold predictions from one split
    _, pb = cvr2(base, 0)
    _, pa = cvr2(base + add, 0)
    e_b, e_a, yy = (x[y] - pb).to_numpy(), (x[y] - pa).to_numpy(), x[y].to_numpy()
    deltas = []
    for _ in range(reps):
        i = RNG.integers(0, len(x), len(x))
        tss = ((yy[i] - yy[i].mean()) ** 2).sum()
        deltas.append(((e_b[i] ** 2).sum() - (e_a[i] ** 2).sum()) / tss)
    return {"outcome": y, "added": " + ".join(add), "n": len(x), "cvR2 base": np.mean(r_b),
            "cvR2 +added": np.mean(r_a), "delta": np.mean(r_a) - np.mean(r_b),
            "delta lo95": np.percentile(deltas, 2.5), "delta hi95": np.percentile(deltas, 97.5)}


def player_level(d, taxes):
    po = player_outcomes(d)
    cr = pd.read_csv(ROOT / "data/raw/combine_results.csv", na_values="NA").rename(columns={"nfl_id": "nflId"})
    meta = d.groupby("nflId").agg(pick=("draftOverallPick", "first"), vmax=("vmax", "first"),
                                  burst2_10=("burst2_10", "first"), name=("displayName", "first"),
                                  pos=("combinePosition", "first"))
    p = (po.join(meta).join(taxes).join(cr.set_index("nflId")[["forty", "combine_weight", "combine_height"]]))
    p["logPick"] = np.log(p.pick.fillna(260))
    p = p[p.routes >= 100]
    base = ["logPick", "forty", "combine_weight", "combine_height"]
    rows = []
    for y in ("sepOE", "targetOE"):
        rows.append(nested_cv(p, y, base, ["vmax", "burst2_10"]))
        rows.append(nested_cv(p, y, base, ["speedTaxLog"]))
        rows.append(nested_cv(p, y, base, ["burstTaxLog"]))
        rows.append(nested_cv(p, y, base, ["speedTaxLog", "burstTaxLog"]))
    # does the Combine predict the game tax? (Combine -> NFL usage link)
    links = []
    for tax in ("speedTaxLog", "burstTaxLog"):
        rows_t = nested_cv(p.assign(**{tax: p[tax]}), tax, ["logPick", "combine_weight", "combine_height"],
                           ["forty", "vmax", "burst2_10"])
        links.append(rows_t)
    corr = p[["sepOE", "targetOE", "speedTaxLog", "burstTaxLog", "vmax", "burst2_10", "forty", "logPick"]].corr(
        method="spearman").round(2)
    return pd.DataFrame(rows), pd.DataFrame(links), corr, p


def beyond_stopwatch(d, players):
    """Player game-speed levels (no Combine term) predicted from tracking capacity beyond pick + size + forty."""
    sp = eligible(d[(d.windowEnd == "pass") & d.peakSpeed.gt(0)], "speedFrac", 40).copy()
    sp["logGame"] = np.log(sp.peakSpeed)
    _, lv_s = mixed(f"logGame ~ {CONTEXT}", sp, "gameSpeedLevel")
    bu = d[(d.windowEnd == "pass") & (d.sAtSnap < 1.0) & (d.motion == 0) & d.burst.gt(0)]
    bu = eligible(bu, "burstFrac", 40).copy()
    bu["logGame"] = np.log(bu.burst)
    _, lv_b = mixed(f"logGame ~ {CONTEXT.replace('bs(tWindow, df=4) + ', '').replace('motion + ', '')}", bu,
                    "gameBurstLevel")
    p = players.join(lv_s).join(lv_b)
    base = ["logPick", "combine_weight", "combine_height", "forty"]
    rows = []
    for y in ("gameSpeedLevel", "gameBurstLevel", "speedTaxLog", "burstTaxLog"):
        for add in (["vmax"], ["burst2_10"], ["vmax", "burst2_10"]):
            rows.append(nested_cv(p, y, base, add))
    return pd.DataFrame(rows)


SUMMARY = """> **Superseded in part (2026-10-16).** These elasticities pool WR and TE, so they mostly reflect WR-vs-TE
> differences. Within WRs only, top speed transfers at about 0.3-0.5 and Combine burst is too unreliable to
> translate (within-WR ICC 0.15). See `docs/translation.md`. The play-level and player-level results below stand.

## Summary (interpretation as of 2026-10-15)

**What holds up**
- **Top speed translates almost one for one.** The elasticity of game peak speed on Combine vmax is 0.82
  [0.61, 1.04]. **Burst translates at about half**: 0.42 [0.21, 0.64], or about 0.5 after correcting for
  Combine retest error. The start from a 3-point stance and a route release are different movements.
- **Within a player, faster reps pay on vertical routes**: +0.10 yd separation per within-player SD
  [0.06, 0.15]. On breaking routes, faster reps are targeted more often (+2.8 pp per SD) with no
  separation change. On short routes, faster reps have *less* separation (-0.18 yd per SD). Receivers who
  settle into a hitch or sit in zone are slower and more open, so the direction there is not causal.
- **The tax is a stable trait** (year-over-year r about 0.87). It is unrelated to draft slot.

**What does not hold up**
- **Combine-relative speed does not beat absolute speed.** Separation is explained as well or better by
  absolute peak speed. A given speed is not worth less to a faster player (vmax term near 0 on vertical
  routes). The "share of capacity" framing adds nothing as an explanatory variable.
- **At the player level, nothing tracking-based adds to draft slot, forty and size**, whether for
  separation over expected, target rate over expected, or even in-game speed and burst levels (all
  ΔR² about 0, n = 75). The one positive, burst2_10 predicting the burst tax, is mechanical: the tax
  is defined relative to burst2_10.

**Implication.** The tracked 40 confirms and sharpens what the stopwatch already says about top speed. Its
burst dimension is reliable at the Combine but largely does not carry over to route releases. A writeup
built on "the share of capacity used predicts production" is not supported. See the reply in the session
log for the options this leaves.

"""


def main() -> int:
    d = prepare(pd.read_parquet(ROUTES))
    d = d[d.vmax.notna()]
    tr, taxes = translation(d)
    within, comp, _ = play_level(d)
    inc, links, corr, players = player_level(d, taxes)
    players.reset_index().to_parquet(PROC / "speed_tax_players_v2.parquet")
    bsw = beyond_stopwatch(d, players)

    DOC.write_text("\n".join([
        "# Speed Tax step 3: payoff tests\n",
        f"_Generated {date.today().isoformat()} by `scripts/speed_tax_payoff.py`._\n",
        SUMMARY,
        "## 1. Combine-to-game translation (elasticities)\n",
        "A mixed model of log(game metric) on log(Combine capacity), with route/window/alignment/coverage/"
        "cushion/field/down context and a player random intercept. β = 1 means a receiver with 10% more Combine "
        "capacity shows 10% more on Sundays. The intercept is the denominator-corrected tax (`*TaxLog`). "
        "Combine burst has retest ICC 0.82, so its β is attenuated by about that factor.\n\n", md(tr),
        "\n## 2a. Within-player payoff: separation and target on speed (player FE, cluster-robust SE)\n",
        "`speedZ` is the rep's peak speed, standardized within player × route (min 5 reps). The coefficient is "
        "the change per within-player SD. Context and player fixed effects are included.\n\n", md(within),
        "\n## 2b. Absolute vs Combine-relative speed (mixed models, ML)\n",
        "Same context and a player random intercept. Positive `logLik - absolute` means Combine-relative speed "
        "explains separation better than absolute speed. `vmaxC` tests whether a given absolute speed is worth "
        "less for a faster player (negative = speed is judged relative to capacity).\n\n", md(comp),
        "\n## 3a. Player level: incremental value beyond draft slot + forty + size (nested CV, routes >= 100)\n",
        "Outcomes are empirical-Bayes-shrunk per-player residuals from a context-only model. Delta CIs are "
        "bootstrapped over players from one out-of-fold split.\n\n", md(inc),
        "\n## 3b. Does the Combine predict the game tax? (base: log pick + size; added: forty, vmax, burst2_10)\n",
        md(links),
        "\n## 3c. Beyond the stopwatch: tracking capacity -> in-game speed (base: log pick + size + official forty)\n",
        "`gameSpeedLevel` and `gameBurstLevel` are player random intercepts from context-only models of log peak "
        "speed and log 2-10 yd burst (absolute, with no Combine term).\n\n", md(bsw),
        "\n## 3d. Player-level Spearman correlations\n", md(corr.reset_index().rename(columns={"index": ""}), 2),
    ]))
    print(DOC.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
