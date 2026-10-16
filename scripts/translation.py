#!/usr/bin/env python
"""Combine-to-game translation of sprint capacity for WR/TE, DB and DL.

    python scripts/translation.py [--rebuild]

For each group, game windows run from the snap to the throw/sack (``bdb27.usage.route_windows``):
WR on routes run (TE excluded: about 34 players), DB on coverage snaps of thrown passes, DL on pass-rush snaps (NGS get-off
recorded). Per group:

1. elasticities: log(game peak speed) ~ log(Combine vmax) and log(game 2-10 yd burst) ~
   log(Combine burst2_10), plus context and a player random intercept, disattenuated for
   Combine measurement error with the group's own retest reliability;
2. beyond the stopwatch: player game-speed and game-burst levels (context-only random intercepts)
   from log pick + size + official forty, plus tracking vmax / burst2_10 (nested CV);
3. play-level payoff within player (player fixed effects): WR/TE separation and target, DL pressure.

Writes data/processed/windows_<group>.parquet, docs/translation.md and docs/fig/translation.png.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from bdb27 import tracking as T  # noqa: E402
from bdb27 import usage as U  # noqa: E402
from speed_tax import md  # noqa: E402
from speed_tax_payoff import nested_cv  # noqa: E402

PROC, RAW = ROOT / "data/processed", ROOT / "data/raw"
DOC, FIG = ROOT / "docs/translation.md", ROOT / "docs/fig/translation.png"
PP_COLS = ["gameId", "playId", "nflId", "routeRan", "linedUpPosition", "teamCoverageManZone", "coverageAssignment",
           "down", "yardsToGo", "target", "separationAtPassForward", "inMotionAtBallSnap", "cushion", "passResult",
           "playerGetOff", "timeToPressure", "timeToQbHurry", "sack"]
GROUPS = {
    "WR": dict(positions=["WR"], filt=pl.col("routeRan").is_not_null()),
    "DB": dict(positions=["DB"], filt=pl.col("coverageAssignment").is_not_null()
               & pl.col("passResult").is_in(["C", "I", "IN"])),
    "DL": dict(positions=["DL"], filt=pl.col("playerGetOff").is_not_null()),
}
BASE_CTX = "C(align) + C(down) + yardsToGo + room + C(season)"
CTX = {
    "WR": f"{BASE_CTX} + C(routeRan) + C(manZone) + cushionF + cushionMissing + motion",
    "DB": f"{BASE_CTX} + C(covAssign) + cushionF + cushionMissing",
    "DL": BASE_CTX,
}
WINDOW = " + bs(tWindow, df=4)"


# ----------------------------------------------------------------------- data
def build(group: str) -> pd.DataFrame:
    spec = GROUPS[group]
    cr = T.load_table("combine_results", RAW)
    ids = cr.filter(pl.col("combinePosition").cast(pl.String).is_in(spec["positions"]))["nflId"].cast(pl.Int32)
    pp = (T.load_table("player_play", RAW).select(PP_COLS).filter(spec["filt"])
          .with_columns(pl.col("nflId").cast(pl.Int32), pl.col("playId").cast(pl.Int32)))
    parts = []
    for season in (2023, 2024, 2025):
        lf = (pl.scan_parquet(PROC / f"game_tracking_{season}_clean.parquet")
              .filter((pl.col("seasonType").cast(pl.String) == "REG") & pl.col("nflId").is_in(ids.implode()))
              .select("gameId", "playId", "nflId", "frameId", "framesFromSnap", "x", "y", "s", "event", "losX"))
        parts.append(U.route_windows(lf).with_columns(season=pl.lit(season, pl.Int16)).join(pp, on=U.KEY, how="inner"))
    w = pl.concat(parts)
    cap = pl.read_parquet(PROC / "capacity_players.parquet").with_columns(pl.col("nflId").cast(pl.Int32))
    w = U.attach_capacity(w, cap)
    out = PROC / f"windows_{group.replace('/', '')}.parquet"
    w.write_parquet(out)
    print(f"  {group}: {w.height} windows, {w['nflId'].n_unique()} players -> {out.name}")
    return w.to_pandas()


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["align"] = d["linedUpPosition"].fillna("UNK")
    d["manZone"] = d["teamCoverageManZone"].fillna("UNK")
    d["covAssign"] = d["coverageAssignment"].fillna("UNK")
    d["room"] = (110 - d["losX"]).clip(0, 100)
    d["motion"] = d["inMotionAtBallSnap"].fillna(False).astype(int)
    d["cushionMissing"] = d["cushion"].isna().astype(int)
    d["cushionF"] = d["cushion"].fillna(d["cushion"].median() if d["cushion"].notna().any() else 0)
    d["pressure"] = (d["timeToPressure"].notna() | d["timeToQbHurry"].notna() | (d["sack"].fillna(0) > 0)).astype(int)
    return d[d["vmax"].notna()]


def eligible(d, col, min_n=40):
    d = d[d[col].notna()]
    return d[d.groupby("nflId")[col].transform("size") >= min_n]


def mixed(formula, d, label, reml=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(formula, d, groups=d["nflId"]).fit(method=["bfgs", "powell"], reml=reml)
    return m, pd.Series({k: v.iloc[0] for k, v in m.random_effects.items()}, name=label)


# ------------------------------------------------------------- reliability of capacity
def capacity_reliability() -> pd.DataFrame:
    """Reliability of each player's log capacity (mean of k reps), by group, from 2-rep retest ICC."""
    att = pd.read_parquet(PROC / "capacity_attempts.parquet")
    att = att[att["usable"]]
    pos = pd.read_csv(RAW / "combine_results.csv", na_values="NA")[["nfl_id", "combine_position"]].rename(
        columns={"nfl_id": "nflId", "combine_position": "pos"})
    att = att.merge(pos, on="nflId")
    att["group"] = att["pos"]
    rows = []
    for g, a in att.groupby("group"):
        k = a.groupby("nflId").size().mean()
        two = (a.sort_values(["nflId", "attempt"]).groupby("nflId").filter(lambda x: len(x) >= 2)
               .groupby("nflId").head(2))  # rows stay player-paired for the reshape below
        for c in ("vmax", "burst2_10"):
            x = np.log(two[c].to_numpy().reshape(-1, 2))
            x = x[~np.isnan(x).any(1)]
            n = len(x)
            msb = 2 * ((x.mean(1) - x.mean()) ** 2).sum() / (n - 1)
            msw = ((x - x.mean(1, keepdims=True)) ** 2).sum() / n
            icc = (msb - msw) / (msb + msw)
            rows.append({"group": g, "capacity": c, "ICC single rep": icc, "mean reps": k,
                         "reliability of player mean": icc * k / (1 + (k - 1) * icc)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ analyses
def elasticities(group, d, rel):
    ctx = CTX[group]
    rows, levels = [], {}
    specs = (("speed", "peakSpeed", "vmax", d[d.windowEnd.isin(["pass", "sack"])], ctx + WINDOW),
             ("burst", "burst", "burst2_10", d[(d.sAtSnap < 1.0) & (d.motion == 0)], ctx.replace(" + motion", "")))
    for name, game, cap, sub, formula in specs:
        x = eligible(sub[sub[game] > 0], game)
        x = x[x[cap].notna()].copy()
        x["logGame"], x["logCapC"] = np.log(x[game]), np.log(x[cap]) - np.log(x[cap]).mean()
        m, _ = mixed(f"logGame ~ logCapC + {formula}", x, f"{name}Tax")
        _, lvl = mixed(f"logGame ~ {formula}", x, f"game{name.title()}Level")
        levels[lvl.name] = lvl
        lam = rel.query("group == @group and capacity == @cap")["reliability of player mean"].iloc[0]
        lo, hi = m.conf_int().loc["logCapC"]
        rows.append({"group": group, "capacity": name, "elasticity": m.params["logCapC"], "lo95": lo, "hi95": hi,
                     "reliability": lam, "disattenuated": m.params["logCapC"] / lam,
                     "game/Combine median": (x[game] / x[cap]).median(), "windows": int(m.nobs),
                     "players": x.nflId.nunique()})
    return rows, pd.DataFrame(levels)


LONG_SUBSETS = {  # windows long enough for the player to approach full speed
    "WR": lambda d: d["routeRan"].isin(["GO", "POST", "CORNER", "WHEEL"]),
    "DB": lambda d: d["covAssign"].str.startswith("DEEP") | (d["covAssign"] == "MAN"),
    "DL": lambda d: pd.Series(True, index=d.index),
}


def speed_sensitivity(group, d):
    """Top-speed elasticity under (a) a Combine weight control and (b) long windows only (>= 3.5 s)."""
    wt = pd.read_csv(RAW / "combine_results.csv", na_values="NA").rename(
        columns={"nfl_id": "nflId"}).set_index("nflId")["combine_weight"]
    base = d[d.windowEnd.isin(["pass", "sack"]) & (d.peakSpeed > 0)].copy()
    base["weightC"] = base["nflId"].map(wt)
    base["weightC"] -= base["weightC"].mean()
    long_ = base[(base.tWindow >= 3.5) & LONG_SUBSETS[group](base)]
    rows = []
    for name, sub, extra, min_n in (("+ Combine weight", base, " + weightC", 40),
                                     ("long windows (>= 3.5 s)", long_, "", 10),
                                     ("long windows + weight", long_, " + weightC", 10)):
        x = eligible(sub, "peakSpeed", min_n).copy()
        x["logGame"], x["logCapC"] = np.log(x.peakSpeed), np.log(x.vmax) - np.log(x.vmax).mean()
        m, _ = mixed(f"logGame ~ logCapC{extra} + {CTX[group]}{WINDOW}", x, "s")
        lo, hi = m.conf_int().loc["logCapC"]
        rows.append({"group": group, "spec": name, "elasticity": m.params["logCapC"], "lo95": lo, "hi95": hi,
                     "game/Combine median": (x.peakSpeed / x.vmax).median(), "windows": int(m.nobs),
                     "players": x.nflId.nunique()})
    return rows


def beyond_stopwatch(group, levels, d):
    cr = pd.read_csv(RAW / "combine_results.csv", na_values="NA").rename(columns={"nfl_id": "nflId"})
    meta = d.groupby("nflId").agg(vmax=("vmax", "first"), burst2_10=("burst2_10", "first"))
    picks = pd.read_csv(RAW / "players.csv", na_values="NA").rename(columns={"nfl_id": "nflId"}).set_index("nflId")
    p = levels.join(meta).join(cr.set_index("nflId")[["forty", "combine_weight", "combine_height"]])
    p["logPick"] = np.log(picks["draft_overall_pick"].reindex(p.index).fillna(260))
    base = ["logPick", "combine_weight", "combine_height", "forty"]
    rows = []
    for y in levels.columns:
        for add in (["vmax"], ["burst2_10"]):
            r = nested_cv(p, y, base, add)
            rows.append({"group": group, **r})
    return rows


def payoff(group, d):
    ctx = CTX[group] + WINDOW
    rows = []
    if group == "WR":
        fam = {"GO": "vertical", "POST": "vertical", "CORNER": "vertical", "WHEEL": "vertical",
               "IN": "breaking", "OUT": "breaking", "CROSS": "breaking", "HITCH": "short", "SLANT": "short", "FLAT": "short"}
        s = d[(d.windowEnd == "pass") & d.separationAtPassForward.notna() & d.routeRan.isin(fam)].copy()
        s["family"], s["targetI"] = s.routeRan.map(fam), s.target.astype(int)
        cell = ["nflId", "routeRan"]
        outcomes = (("separationAtPassForward", "separation (yd)"), ("targetI", "target (prob)"))
        splits = [("vertical", s.family == "vertical"), ("breaking", s.family == "breaking"), ("short", s.family == "short")]
    elif group == "DL":
        s = d[d.windowEnd.isin(["pass", "sack"])].copy()
        cell = ["nflId", "align"]
        outcomes = (("pressure", "pressure (prob)"),)
        splits = [("EDGE", s["align"] == "EDGE"), ("INTERIOR", s["align"] == "INTERIOR_LINE")]
    else:
        return rows
    metrics = ("peakSpeed", "burst") + (("quickGetOff",) if group == "DL" else ())
    if group == "DL":
        s["quickGetOff"] = -s["playerGetOff"]  # NGS get-off time, negated: higher = quicker first step
    for metric in metrics:
        x0 = s[s[metric].notna()].copy()
        g = x0.groupby(cell)[metric]
        x0["z"] = (x0[metric] - g.transform("mean")) / g.transform("std")
        x0 = x0[g.transform("size") >= 5].dropna(subset=["z"])
        for sname, mask in splits:
            x = x0[mask.reindex(x0.index, fill_value=False)]
            for y, lab in outcomes:
                m = smf.ols(f"{y} ~ z + {ctx} + C(nflId)", x).fit(cov_type="cluster", cov_kwds={"groups": x["nflId"]})
                lo, hi = m.conf_int().loc["z"]
                rows.append({"group": group, "subset": sname, "rep metric (within-player z)": metric, "outcome": lab,
                             "coef per SD": m.params["z"], "lo95": lo, "hi95": hi, "p": m.pvalues["z"],
                             "n": int(m.nobs), "players": x.nflId.nunique()})
    return rows


def figure(el: pd.DataFrame, sens: pd.DataFrame):
    """Top-speed elasticity by group under three windows: snap-to-throw, long windows, whole-play p99.9."""
    FIG.parent.mkdir(parents=True, exist_ok=True)
    mx_path = PROC / "eda/max_speed_elasticity.csv"
    mx = pd.read_csv(mx_path) if mx_path.exists() else pd.DataFrame()
    specs = [("snap to throw, all reps", "#9aa5b1"), ("long reps (>= 3.5 s)", "#2a6fdb"),
             ("whole play, fastest 0.1%", "#e67e22")]
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    groups = list(GROUPS)
    for i, g in enumerate(groups):
        vals = []
        r = sens[(sens.group == g) & (sens.spec == "+ Combine weight")]
        vals.append(r.iloc[0][["elasticity", "lo95", "hi95"]].to_numpy() if len(r) else None)
        r = sens[(sens.group == g) & (sens.spec == "long windows + weight")]
        vals.append(r.iloc[0][["elasticity", "lo95", "hi95"]].to_numpy() if len(r) else None)
        r = mx[(mx.get("group") == g) & (mx.get("game speed") == "p999") & (mx.get("weight ctrl") == True)] if len(mx) else mx  # noqa: E712
        vals.append(r.iloc[0][["elasticity", "lo95", "hi95"]].to_numpy().astype(float) if len(r) else None)
        for j, ((lab, col), v) in enumerate(zip(specs, vals)):
            if v is None:
                continue
            y = i + (j - 1) * 0.22
            ax.plot([v[1], v[2]], [y, y], color=col, lw=2.5, solid_capstyle="round")
            ax.plot(v[0], y, "o", color=col, ms=7)
            if i == 0:
                ax.text(v[2] + 0.03, y, lab, color=col, va="center", fontsize=8.5)
    ax.axvline(1, color="#555", lw=1, ls="--")
    ax.text(1.02, -0.55, "full transfer", fontsize=8, color="#555", ha="left")
    ax.axvline(0, color="#ccc", lw=1)
    ax.set_yticks(range(len(groups)), groups)
    ax.invert_yaxis()
    ax.set_xlim(-0.3, 1.6)
    ax.set_xlabel("Elasticity of game speed on Combine top speed (within group, weight-controlled, 95% CI)")
    ax.set_title("Forty cents on the dollar: Combine speed edges shrink on Sundays")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)


SUMMARY = """## Summary (2026-10-16)

- **Forty cents on the dollar.** Within a position group and holding body weight fixed, a Combine top-speed
  edge shows up in game speed at an elasticity of about **0.3-0.5**: WR 0.27 overall and 0.38 on long vertical
  routes; DB 0.38-0.47; DL 0.42-0.45. A WR who is 5% faster at the Combine is about 2% faster on Sundays.
- **The shrinkage is not window truncation.** Whole-play maximum speed (fastest 0.1% of frames, including after
  the catch and in pursuit) gives WR 0.29-0.36, DB 0.38 and DL 0.62 with weight controlled. At that top end, WRs
  reach only about 84% of their Combine top speed.
- **Pooling groups overstates transfer.** WR+TE pooled gives 0.82, and DL without a weight control gives 1.0.
  Those mostly reflect body type and role (TE vs WR, DT vs edge), not translation.
- **Combine burst cannot be translated for WRs.** Its within-WR retest ICC is only 0.15. DL burst is
  reliable (0.81) and transfers at about 0.6 (disattenuated), but adds nothing beyond the official forty
  at the player level.
- **Beyond the stopwatch: no.** For every group, tracking vmax/burst adds about 0 to log pick + size + forty
  when predicting a player's game speed or burst level.
- **Faster reps still pay, within player.** WR vertical routes: +0.10 yd separation per SD; breaking routes:
  +2.8 pp target probability. DL: a quicker get-off gives +2.6 to 3.2 pp pressure per SD (the whole-rush speed
  coefficients are inflated by reverse causality).

"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    rel = capacity_reliability()
    el_rows, bsw_rows, pay_rows, counts, sens_rows = [], [], [], [], []
    for group in GROUPS:
        path = PROC / f"windows_{group.replace('/', '')}.parquet"
        raw = build(group) if args.rebuild or not path.exists() else pd.read_parquet(path)
        d = prepare(raw)
        counts.append({"group": group, "windows": len(raw), "players": raw.nflId.nunique(),
                       "with capacity": d.nflId.nunique(), "median window (s)": d.tWindow.median(),
                       "reach 10 yd of path": d.burst.notna().mean()})
        rows, levels = elasticities(group, d, rel)
        el_rows += rows
        sens_rows += speed_sensitivity(group, d)
        bsw_rows += beyond_stopwatch(group, levels, d)
        pay_rows += payoff(group, d)
        print(f"  {group} done")
    el, bsw, pay = pd.DataFrame(el_rows), pd.DataFrame(bsw_rows), pd.DataFrame(pay_rows)
    sens = pd.DataFrame(sens_rows)
    figure(el, sens)
    DOC.write_text("\n".join([
        "# Combine-to-game translation of sprint capacity\n",
        f"_Generated {date.today().isoformat()} by `scripts/translation.py`._\n",
        SUMMARY,
        "Game windows run from the snap to the throw or sack. WRs are measured on routes run (TEs are excluded, with too few players), DB on coverage "
        "snaps of thrown passes, DL on pass-rush snaps (NGS get-off recorded). All are REG only. Capacities come from "
        "the Combine 40 (`docs/capacity.md`). Burst is mean path speed from 2 to 10 yd; in games it is only measured "
        "with no speed at the snap and no motion.\n",
        "## Samples\n", md(pd.DataFrame(counts)),
        "\n## Reliability of the Combine capacity (log scale)\n", md(rel),
        "\n## 1. Elasticities: log(game) ~ log(Combine) + context + (1 | player)\n",
        "An elasticity of 1 means Combine differences between players show up fully in games. "
        "`disattenuated` divides by the reliability of the player's Combine mean.\n\n", md(el),
        "\n![translation](fig/translation.png)\n",
        "\n### Top-speed sensitivity: body weight and long windows\n",
        "The long windows test whether short snap-to-throw windows compress speed differences: WR vertical "
        "routes, DB deep or man coverage, DL all rushes, each lasting at least 3.5 s.\n\n",
        md(sens),
        "\nWhole-play maximum speed (fastest 1% / 0.1% of each player's REG frames, including after the catch and in "
        "pursuit) gives the same picture: `scripts/eda/max_speed_elasticity.py`, "
        "`data/processed/eda/max_speed_elasticity.csv`.\n",
        "\n## 2. Beyond the stopwatch: game speed/burst level from log pick + size + official forty, then + tracking\n",
        md(bsw),
        "\n## 3. Within-player payoff of faster reps (player FE, cluster-robust SE; z within player × route/alignment)\n",
        md(pay),
        "\n**Caveat (DL).** Peak speed and 2-10 yd burst are measured over the whole rush, so they partly *result* "
        "from beating the block: free rushers accelerate toward the QB. Their pressure coefficients overstate the causal "
        "payoff. `quickGetOff` (negated NGS get-off, the first steps before contact) is the cleaner rep measure. "
        "For short WR routes, the negative separation coefficients reflect receivers settling into soft spots "
        "(slower and more open), not a cost of speed.\n",
    ]))
    print(DOC.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
