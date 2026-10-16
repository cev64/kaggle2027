#!/usr/bin/env python
"""Sunday Speed: a scout-facing projection of NFL game top speed from Combine sprint data.

    PYTHONPATH=src:scripts python scripts/sunday_speed.py

Steps (WR, DB, DL; REG season only):

1. Game-speed target per player. Two candidates, compared by split-half reliability (alternating
   games within player, Spearman-Brown): quantiles of per-play peak speed (chosen: 90th pct), frame-level
   p99 / p99.9, and the player random
   intercept of the context-adjusted route/rush window model from ``scripts/translation.py``.
2. Projection models of the target from Combine information, validated out of sample by
   leave-one-draft-class-out and repeated 5-fold CV: naive full transfer (elasticity 1), group mean,
   shrunk (estimated elasticity + weight), official forty + weight, tracking vmax + forty + weight.
3. 80% prediction intervals (OLS on the log scale), with their out-of-sample coverage.
4. Football units (mph) and an illustrative vertical-route separation value; 2025-class scout table;
   biggest misses of naive full transfer; pre-draft view for every prospect with a tracked 40.

Writes data/processed/sunday_speed_players.parquet, docs/sunday_speed.md and docs/fig/sunday_speed.png.
"""
from __future__ import annotations

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
from scipy import stats  # noqa: E402

from speed_tax import md  # noqa: E402
from translation import CTX, WINDOW, eligible, mixed, prepare  # noqa: E402

PROC, RAW = ROOT / "data/processed", ROOT / "data/raw"
DOC, FIG = ROOT / "docs/sunday_speed.md", ROOT / "docs/fig/sunday_speed.png"
OUT = PROC / "sunday_speed_players.parquet"
GROUPS = ("WR", "DB", "DL")
MIN_PLAYS = 200          # whole-play target: REG plays on the field
MIN_WIN = 40             # window target: windows per player (20 per half for split-half)
MPH = 3600 / 1760        # yd/s -> mph
SEP_PER_SD = 0.106       # WR vertical routes, within-player separation per SD of rep peak speed (translation.md)
VERTICAL = ["GO", "POST", "CORNER", "WHEEL"]
BLUE, VERM, GREY = "#0072B2", "#D55E00", "#9a9a9a"   # Okabe-Ito, validated CVD-safe
RNG = np.random.default_rng(2027)


# ------------------------------------------------------------------ 1. game targets
def load_capacity() -> pd.DataFrame:
    cap = pd.read_parquet(PROC / "capacity_players.parquet")
    cap["displayName"] = cap["displayName"].astype(str)
    names = pd.read_csv(RAW / "players.csv", na_values="NA")[["nfl_id", "display_name"]]
    cap = cap.merge(names.rename(columns={"nfl_id": "nflId"}), on="nflId", how="left")
    cap["name"] = cap["display_name"].fillna(cap["displayName"])
    return cap[cap.combinePosition.isin(GROUPS)].drop(columns="display_name")


def frame_speeds(ids) -> pl.DataFrame:
    """nflId, gameId, playId, s for REG frames of the given players (about 10M rows, < 0.5 GB)."""
    parts = []
    for season in (2023, 2024, 2025):
        parts.append(pl.scan_parquet(PROC / f"game_tracking_{season}_clean.parquet")
                     .filter((pl.col("seasonType").cast(pl.String) == "REG")
                             & pl.col("nflId").is_in(pl.Series(ids, dtype=pl.Int32).implode()))
                     .select("nflId", "gameId", "playId", "s").collect())
    return pl.concat(parts)


def half_of(keys: pd.DataFrame) -> pd.Series:
    """Alternate games within player (by gameId order): 0 / 1."""
    return keys.groupby("nflId")["gameId"].rank(method="dense").astype(int) % 2


def whole_play_targets(ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-play peak speed (all REG plays, whole play) -> player quantiles; also frame-level p99 / p99.9."""
    fr = frame_speeds(ids)
    games = fr.select("nflId", "gameId").unique().to_pandas()
    games["half"] = half_of(games)
    fr = fr.join(pl.from_pandas(games).with_columns(pl.col("nflId").cast(pl.Int32), pl.col("half").cast(pl.Int8)),
                 on=["nflId", "gameId"])
    peaks = fr.group_by("nflId", "half", "gameId", "playId").agg(pk=pl.col("s").max())
    fq = dict(p99=pl.col("s").quantile(0.99), p999=pl.col("s").quantile(0.999))
    pq = dict(q80=pl.col("pk").quantile(0.80), q90=pl.col("pk").quantile(0.90), plays=pl.len(),
              games=pl.col("gameId").n_unique())
    full = (fr.group_by("nflId").agg(**fq).join(peaks.group_by("nflId").agg(**pq), on="nflId")
            .to_pandas().set_index("nflId"))
    half = fr.group_by("nflId", "half").agg(**fq).join(peaks.group_by("nflId", "half").agg(**pq),
                                                       on=["nflId", "half"]).to_pandas()
    del fr, peaks
    return full, half


def window_targets(group: str):
    """Player random intercept of log window peak speed, context + window length adjusted, in yd/s."""
    d = prepare(pd.read_parquet(PROC / f"windows_{group}.parquet"))
    d = d[d.windowEnd.isin(["pass", "sack"]) & (d.peakSpeed > 0)].copy()
    d["logGame"] = np.log(d.peakSpeed)
    d["half"] = half_of(d[["nflId", "gameId"]])
    f = f"logGame ~ {CTX[group]}{WINDOW}"

    def level(x, label):
        m, re = mixed(f, x, label)
        mu = float(np.mean(m.model.exog @ m.fe_params.to_numpy()))
        return np.exp(mu + re).rename(label), m

    full, m = level(eligible(d, "peakSpeed", MIN_WIN), "winLevel")
    halves = {}
    for h in (0, 1):
        x = d[d.half == h]
        halves[h], _ = level(eligible(x, "peakSpeed", MIN_WIN // 2), f"h{h}")
    hv = pd.concat(halves.values(), axis=1).dropna()
    # vertical-route rep speed, for the separation illustration (WR only)
    vert = None
    if group == "WR":
        v = d[d.routeRan.isin(VERTICAL) & (d.windowEnd == "pass")]
        g = v.groupby(["nflId", "routeRan"])["peakSpeed"]
        sd_within = float(np.sqrt((v.peakSpeed - g.transform("mean")).pow(2).sum() / (len(v) - g.ngroups)))
        vert = dict(sd_within=sd_within, playerMean=v.groupby("nflId")["peakSpeed"].mean(),
                    n=v.groupby("nflId").size())
    return full, hv, vert


TARGETS = {"q90": "per-play peak, 90th pct (chosen)", "q80": "per-play peak, 80th pct",
           "p99": "frame speed p99", "p999": "frame speed p99.9"}
TARGET = "q90"


def split_half(a, b) -> dict:
    r = np.corrcoef(a, b)[0, 1]
    return {"players": len(a), "split-half r": r, "Spearman-Brown reliability": 2 * r / (1 + r)}


# ------------------------------------------------------------------ 2. projection models
def design(df, cols):
    return np.column_stack([np.ones(len(df))] + [np.log(df[c].to_numpy(float)) for c in cols])


LINEAR = {  # name -> log predictors
    "(b2) weight only": ["combineWeight"],
    "(c0) elasticity only": ["vmax"],
    "(c) shrunk: vmax + weight": ["vmax", "combineWeight"],
    "(d) official forty + weight": ["forty", "combineWeight"],
    "(e) vmax + forty + weight": ["vmax", "forty", "combineWeight"],
}
MODELS = ["(a) naive full transfer", "(b) group mean"] + list(LINEAR)


def fit_predict(tr, te, y="target", level=0.80):
    """Predictions (yd/s) for every model, plus the 80% PI of the shrunk model."""
    out = pd.DataFrame(index=te.index)
    ly = np.log(tr[y].to_numpy())
    out["(a) naive full transfer"] = np.exp(np.mean(ly - np.log(tr.vmax))) * te.vmax
    out["(b) group mean"] = tr[y].mean()
    for name, cols in LINEAR.items():
        X, X0 = design(tr, cols), design(te, cols)
        beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
        out[name] = np.exp(X0 @ beta)
        if name.startswith("(c)"):
            df = len(ly) - X.shape[1]
            s2 = np.sum((ly - X @ beta) ** 2) / df
            se = np.sqrt(s2 * (1 + np.einsum("ij,jk,ik->i", X0, np.linalg.inv(X.T @ X), X0)))
            q = stats.t.ppf(0.5 + level / 2, df)
            out["lo80"], out["hi80"] = np.exp(X0 @ beta - q * se), np.exp(X0 @ beta + q * se)
            out["elasticity"] = beta[1]
    return out


def scores(obs, pred, groups=None, model="") -> dict:
    """RMSE, MAE, calibration slope (observed on predicted; within group when ``groups`` is given)."""
    e = obs - pred
    if groups is not None:  # pooled groups: centre both on the group's observed mean
        gm = obs.groupby(groups).transform("mean")
        obs, pred = obs - gm, pred - gm
    slope = np.polyfit(pred, obs, 1)[0] if not model.startswith("(b)") else np.nan
    return {"RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)), "calibration slope": slope}


def lodo(d: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for g, dg in d.groupby("group"):
        for yr in sorted(dg.draftYear.unique()):
            te, tr = dg[dg.draftYear == yr], dg[dg.draftYear != yr]
            parts.append(fit_predict(tr, te))
    return pd.concat(parts)


def repeated_kfold(d: pd.DataFrame, k=5, reps=40) -> pd.DataFrame:
    rows = []
    for g, dg in d.groupby("group"):
        for r in range(reps):
            fold = RNG.permutation(np.arange(len(dg)) % k)
            pred = pd.concat([fit_predict(dg[fold != f], dg[fold == f]) for f in range(k)]).loc[dg.index]
            for m in MODELS:
                rows.append({"group": g, "rep": r, "model": m, **scores(dg.target, pred[m], model=m)})
    return pd.DataFrame(rows)


def boot_diff(obs, p1, p2, n=4000):
    """RMSE(p1) - RMSE(p2), percentile bootstrap over players."""
    e1, e2 = (obs - p1).to_numpy() ** 2, (obs - p2).to_numpy() ** 2
    idx = RNG.integers(0, len(e1), (n, len(e1)))
    b = np.sqrt(e1[idx].mean(1)) - np.sqrt(e2[idx].mean(1))
    return np.sqrt(e1.mean()) - np.sqrt(e2.mean()), *np.percentile(b, [2.5, 97.5])


# ------------------------------------------------------------------ figure
def figure(d, full_fits, notable):
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, g in zip(axes, GROUPS):
        x = d[d.group == g]
        ax.scatter(x.vmax, x.target, s=34, color=GREY, alpha=0.75, edgecolor="white", lw=0.6, zorder=2)
        grid = np.linspace(x.vmax.min(), x.vmax.max(), 50)
        ratio = np.exp(np.mean(np.log(x.target) - np.log(x.vmax)))
        b = full_fits[g]
        shr = np.exp(b[0] + b[1] * np.log(grid) + b[2] * np.mean(np.log(x.combineWeight)))
        ax.plot(grid, ratio * grid, color=VERM, lw=2.5, ls="--", zorder=3)
        ax.plot(grid, shr, color=BLUE, lw=2.5, zorder=3)
        ax.text(grid[-1], ratio * grid[-1], "  naive\n  (elasticity 1)", color=VERM, va="center", fontsize=12)
        ax.text(grid[-1], shr[-1], f"  shrunk\n  ({b[1]:.2f})", color=BLUE, va="center", fontsize=12)
        for _, r in notable[notable.group == g].iterrows():
            ax.scatter([r.vmax], [r.target], s=60, facecolor="none", edgecolor="#222", lw=1.4, zorder=4)
            ax.annotate(f"{r['name']}\n(tests fast, plays slower)", (r.vmax, r.target), xytext=(-10, 4),
                        textcoords="offset points", ha="right", va="bottom", fontsize=10.5, color="#222",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=5)
        ax.set_title(g, fontsize=15, loc="left", fontweight="bold")
        ax.set_xlabel("Combine top speed, tracked 40 (yd/s)")
        ax.set_xlim(x.vmax.min() - 0.1, x.vmax.max() + 0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Sunday top speed: 90th pct of play peaks (yd/s)")
    fig.suptitle("Sunday Speed: Combine edges shrink in games. Naive full transfer over-spreads players\n"
                 "Shrunk line: within-group fit at the group's mean weight (for DL, weight, not 40 speed, sorts game speed)",
                 fontsize=14, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    plt.close(fig)


SUMMARY = """## Summary

- **Target: "Sunday top speed" = the 90th percentile of a player's per-play peak speed** (all REG plays, whole play,
  >= {v[minp]} plays). It is the speed he reaches on his fastest one-in-ten plays. Split-half reliability
  (alternating games, Spearman-Brown) is WR {v[rq_WR]:.2f}, DB {v[rq_DB]:.2f}, DL {v[rq_DL]:.2f}. That beats frame-level
  p99 (DB {v[rp_DB]:.2f}) and is comparable to the context-adjusted window intercept (WR {v[rw_WR]:.2f}, DB {v[rw_DB]:.2f},
  DL {v[rw_DL]:.2f}), and it reads directly as a top speed in yd/s / mph. The window intercept is kept as a robustness target.
- **Shrunk beats naive full transfer by a clear margin, in every group.** Leave-one-draft-class-out RMSE: naive
  {v[n_all]:.3f} vs shrunk {v[s_all]:.3f} yd/s (difference {v[d_all]:.3f}, 95% CI {v[dl_all]:.3f} to {v[dh_all]:.3f};
  {v[pct_all]:.0f}% lower). It holds for WR, DB and DL separately and on the window target. Naive calibration slopes are
  WR {v[cn_WR]:.2f}, DB {v[cn_DB]:.2f}, DL {v[cn_DL]:.2f}: taking the stopwatch at face value spreads players out 1.5-2x
  too much (DB: no signal at all). Shrunk slopes are WR {v[cs_WR]:.2f}, DB {v[cs_DB]:.2f}, DL {v[cs_DL]:.2f}.
- **Most of the gain is shrinkage to body type, not speed.** Beyond weight, Combine speed improves the projection only
  for WR (weight-only minus shrunk RMSE {v[w_WR]:+.3f} yd/s, CI {v[wl_WR]:.3f} to {v[wh_WR]:.3f}; elasticity
  {v[e_WR]:.2f}). For DB (elasticity {v[e_DB]:.2f}) and DL ({v[e_DL]:.2f}) on this whole-play target, weight carries the
  signal: the shrunk model is no better than weight alone (DL: slightly worse, {v[w_DL]:+.3f} yd/s). On the
  route/rush-window target, the weight-controlled elasticities are WR {v[ew_WR]:.2f}, DB {v[ew_DB]:.2f} and DL {v[ew_DL]:.2f},
  consistent with "forty cents on the dollar".
- **Stopwatch equivalent: yes.** Official forty + weight matches tracking vmax + weight (pooled RMSE difference
  {v[f_all]:+.3f} yd/s, CI {v[fl_all]:.3f} to {v[fh_all]:.3f}), and adding vmax to the forty adds nothing.
- **Uncertainty.** The 80% prediction interval is about ±{v[pi]:.1f} mph. Held-out coverage is {v[cov]:.0%}
  (WR {v[cov_WR]:.0%}, DB {v[cov_DB]:.0%}, DL {v[cov_DL]:.0%}), so the bands are honest.
"""
HOW = """## How a scout uses this

Do not read a prospect's Combine top speed as his game speed. Read the **projected Sunday speed and its 80% band**
(group + weight + tracked vmax, or official forty, which works equally well). For receivers, 1 mph of extra Combine
top speed buys about {v[e_WR]:.1f} mph on Sundays. For DBs and DL, the band is mostly set by position and body
weight, so a fast 40 should not move the speed projection. Two prospects whose bands overlap (about ±{v[pi]:.1f} mph)
are the same speed for planning purposes. The naive column is what "trusting the stopwatch" implies. The players
it misses most (section 4) are where the scout should look at film for the reason: role, injury, or effort.
The separation column (section 5) is a rough illustration, not a validated effect.
"""


# ------------------------------------------------------------------ main
def main() -> int:
    cap = load_capacity()
    print("frames ...")
    wp, wp_half = whole_play_targets(cap.nflId.to_numpy())
    rel_rows, win_levels, vert = [], {}, None
    for g in GROUPS:
        ids = set(cap.loc[cap.combinePosition == g, "nflId"])
        print(f"window model {g} ...")
        lvl, hv, v = window_targets(g)
        win_levels[g] = lvl
        if v is not None:
            vert = v
        rel_rows.append({"group": g, "target": "window random intercept (context-adjusted)",
                         **split_half(hv.h0, hv.h1)})
        h = wp_half[wp_half.nflId.isin(ids) & (wp_half.plays >= MIN_PLAYS // 2)]
        for q, lab in TARGETS.items():
            w = h.pivot(index="nflId", columns="half", values=q).dropna()
            rel_rows.append({"group": g, "target": lab, **split_half(w[0], w[1])})
    rel = pd.DataFrame(rel_rows)
    print(rel)

    # ---- player table
    p = cap.set_index("nflId")
    p = p.join(wp[list(TARGETS) + ["plays", "games"]])
    p = p.join(pd.concat(win_levels.values()).rename("winLevel"))
    p["group"] = p.combinePosition
    p["target"] = p[TARGET].where(p.plays >= MIN_PLAYS)
    ok = p.target.notna() & (p.vmaxSource == "tracked") & p.forty.notna() & p.combineWeight.notna()
    d = p[ok].copy()
    # how the two targets agree
    agree = d.groupby("group").apply(lambda x: x[["target", "winLevel"]].corr().iloc[0, 1]).rename("r")

    # ---- out-of-sample comparisons
    lo = lodo(d).loc[d.index]
    rows = []
    for g in GROUPS + ("all",):
        m = d.group == g if g != "all" else d.group.notna()
        for name in MODELS:
            rows.append({"group": g, "model": name, "n": int(m.sum()), **scores(d.target[m], lo.loc[m, name], d.group[m] if g == "all" else None, name)})
    lodo_tab = pd.DataFrame(rows)
    cover = d.target.between(lo.lo80, lo.hi80)
    kf = repeated_kfold(d)
    # robustness: same comparison on the context-adjusted window level (route / rush speed, yd/s)
    okw = p.winLevel.notna() & (p.vmaxSource == "tracked") & p.forty.notna() & p.combineWeight.notna()
    dw = p[okw].assign(target=p.winLevel)
    low = lodo(dw).loc[dw.index]
    rob = []
    for g in GROUPS:
        m = dw.group == g
        for name in ["(a) naive full transfer", "(b) group mean", "(b2) weight only", "(c) shrunk: vmax + weight",
                     "(d) official forty + weight"]:
            rob.append({"group": g, "model": name, "n": int(m.sum()), **scores(dw.target[m], low.loc[m, name], model=name)})
        est, l_, h_ = boot_diff(dw.target[m], low.loc[m, "(a) naive full transfer"], low.loc[m, "(c) shrunk: vmax + weight"])
        rob.append({"group": g, "model": "naive - shrunk RMSE [95% CI]", "n": int(m.sum()), "RMSE": est,
                    "MAE": np.nan, "calibration slope": np.nan, "CI": f"[{l_:.3f}, {h_:.3f}]"})
        X = design(dw[m], LINEAR["(c) shrunk: vmax + weight"])
        rob.append({"group": g, "model": "elasticity (c), full sample", "n": int(m.sum()),
                    "RMSE": np.linalg.lstsq(X, np.log(dw.target[m].to_numpy()), rcond=None)[0][1]})
    rob = pd.DataFrame(rob)
    rob["CI"] = rob["CI"].fillna("")
    kf_tab = kf.groupby(["group", "model"], sort=False)[["RMSE", "MAE", "calibration slope"]].mean().reset_index()

    contrasts = []
    pairs = [("(a) naive full transfer", "(c) shrunk: vmax + weight"), ("(b) group mean", "(c) shrunk: vmax + weight"),
             ("(b2) weight only", "(c) shrunk: vmax + weight"),
             ("(d) official forty + weight", "(c) shrunk: vmax + weight"),
             ("(d) official forty + weight", "(e) vmax + forty + weight")]
    for g in GROUPS + ("all",):
        m = d.group == g if g != "all" else d.group.notna()
        for a, b in pairs:
            est, l, h = boot_diff(d.target[m], lo.loc[m, a], lo.loc[m, b])
            contrasts.append({"group": g, "RMSE(A) - RMSE(B)": f"{a} - {b}", "yd/s": est, "lo95": l, "hi95": h,
                              "% of A": 100 * est / np.sqrt(np.mean((d.target[m] - lo.loc[m, a]) ** 2))})
    con = pd.DataFrame(contrasts)

    # ---- full-sample fits, pre-draft projections for every prospect with a tracked 40
    full_fits, proj = {}, []
    pre = p[(p.vmaxSource == "tracked") & p.combineWeight.notna()].copy()
    for g in GROUPS:
        dg = d[d.group == g]
        X = design(dg, LINEAR["(c) shrunk: vmax + weight"])
        full_fits[g], *_ = np.linalg.lstsq(X, np.log(dg.target.to_numpy()), rcond=None)
        proj.append(fit_predict(dg, pre[pre.group == g]))
    proj = pd.concat(proj)
    pre["projSunday"] = proj["(c) shrunk: vmax + weight"]
    pre["projLo80"], pre["projHi80"] = proj.lo80, proj.hi80
    pre["naiveSunday"] = proj["(a) naive full transfer"]
    pre["projFortyModel"] = proj["(d) official forty + weight"]
    for c in ("projSunday", "projLo80", "projHi80", "naiveSunday"):
        pre[c.replace("Sunday", "SundayMph").replace("80", "80Mph") if "Sunday" in c else c + "Mph"] = pre[c] * MPH
    # held-out (LODO) predictions for players with a target
    for c, src in (("lodoShrunk", "(c) shrunk: vmax + weight"), ("lodoNaive", "(a) naive full transfer"),
                   ("lodoForty", "(d) official forty + weight"), ("lodoGroupMean", "(b) group mean"),
                   ("lodoLo80", "lo80"), ("lodoHi80", "hi80")):
        pre[c] = lo[src].reindex(pre.index)

    # ---- separation illustration (WR): projected edge vs group mean -> vertical-route rep speed -> separation
    wr = d[d.group == "WR"].join(vert["playerMean"].rename("vertMean")).join(vert["n"].rename("vertN"))
    wr = wr[wr.vertN >= 20]
    k = np.polyfit(wr.target, wr.vertMean, 1)[0]   # yd/s of vertical-route peak per yd/s of target
    sep_per_yds = SEP_PER_SD * k / vert["sd_within"]
    wrm = pre.group == "WR"
    for c, src in (("sepEdgeShrunk", "projSunday"), ("sepEdgeNaive", "naiveSunday")):
        pre[c] = np.where(wrm, sep_per_yds * (pre[src] - pre.loc[wrm, src].mean()), np.nan)
    pre["hasTarget"] = pre.index.isin(d.index)
    pre["actualSunday"] = pre.target.where(pre.hasTarget)
    pre["actualSundayMph"] = pre.actualSunday * MPH
    keep = ["name", "group", "draftYear", "draftOverallPick", "combineWeight", "forty", "vmax", "plays", "games",
            "hasTarget", "actualSunday", "actualSundayMph", "winLevel", "naiveSunday", "naiveSundayMph", "projSunday",
            "projSundayMph", "projLo80", "projHi80", "projLo80Mph", "projHi80Mph", "projFortyModel",
            "lodoShrunk", "lodoNaive", "lodoForty", "lodoGroupMean", "lodoLo80", "lodoHi80", "sepEdgeShrunk", "sepEdgeNaive"]
    pre = pre.reset_index()[["nflId"] + keep]
    pre.to_parquet(OUT, index=False)

    # ---- scout tables
    t = pre[pre.hasTarget].copy()
    t["missNaive"], t["missShrunk"] = t.actualSunday - t.lodoNaive, t.actualSunday - t.lodoShrunk
    t["in80"] = t.actualSunday.between(t.lodoLo80, t.lodoHi80)

    def fmt(x, cols):
        x = x.copy()
        x["pick"] = x.draftOverallPick.map(lambda v: "UDFA" if pd.isna(v) else f"{v:.0f}")
        x["vmax (mph)"] = (x.vmax * MPH).round(1)
        x["naive (mph)"] = (x.lodoNaive * MPH).round(1)
        x["Sunday proj (mph) [80%]"] = [f"{a * MPH:.1f} [{lo_ * MPH:.1f}, {hi * MPH:.1f}]"
                                        for a, lo_, hi in zip(x.lodoShrunk, x.lodoLo80, x.lodoHi80)]
        x["actual (mph)"] = (x.actualSunday * MPH).round(1)
        x["miss naive (mph)"] = (x.missNaive * MPH).round(2)
        x["miss shrunk (mph)"] = (x.missShrunk * MPH).round(2)
        return x[cols]

    base_cols = ["name", "group", "pick", "forty", "vmax (mph)", "naive (mph)", "Sunday proj (mph) [80%]",
                 "actual (mph)"]
    c25 = t[t.draftYear == 2025].sort_values(["group", "draftOverallPick"], na_position="last")
    c25_tab = fmt(c25.groupby("group", group_keys=False).head(8), base_cols + ["miss shrunk (mph)"])
    over = fmt(t.nsmallest(8, "missNaive"), base_cols + ["draftYear", "miss naive (mph)", "miss shrunk (mph)"])
    under = fmt(t.nlargest(8, "missNaive"), base_cols + ["draftYear", "miss naive (mph)", "miss shrunk (mph)"])

    # pre-draft view: no / thin NFL tracking
    thin = pre[~pre.hasTarget].sort_values("draftOverallPick", na_position="last").copy()
    thin["pick"] = thin.draftOverallPick.map(lambda v: "UDFA" if pd.isna(v) else f"{v:.0f}")
    thin["REG plays"] = thin.plays.fillna(0).astype(int)
    thin["vmax (mph)"] = (thin.vmax * MPH).round(1)
    thin["Sunday proj (mph) [80%]"] = [f"{a:.1f} [{lo_:.1f}, {hi:.1f}]" for a, lo_, hi in
                                       zip(thin.projSundayMph, thin.projLo80Mph, thin.projHi80Mph)]
    thin["naive (mph)"] = thin.naiveSundayMph.round(1)
    thin["sep. edge, illustrative (yd)"] = thin.sepEdgeShrunk.round(3)
    thin_tab = thin.head(12)[["name", "group", "draftYear", "pick", "REG plays", "forty", "vmax (mph)", "naive (mph)",
                              "Sunday proj (mph) [80%]", "sep. edge, illustrative (yd)"]]

    # separation illustration rows: fastest vs slowest tracked WR (by projection)
    wrp = pre[pre.group == "WR"]
    sep_rows = []
    for lab, r in (("fastest WR vmax", wrp.loc[wrp.vmax.idxmax()]), ("slowest WR vmax", wrp.loc[wrp.vmax.idxmin()])):
        sep_rows.append({"player": f"{lab}: {r['name']}", "vmax (mph)": r.vmax * MPH,
                         "naive proj (mph)": r.naiveSundayMph, "shrunk proj (mph)": r.projSundayMph,
                         "sep. edge naive (yd)": r.sepEdgeNaive, "sep. edge shrunk (yd)": r.sepEdgeShrunk})
    sep_tab = pd.DataFrame(sep_rows)

    notable = t.sort_values("missNaive").groupby("group").head(1)   # biggest "tests fast, plays slower" per group
    notable = d.loc[notable.nflId].assign(name=notable.name.to_numpy())
    figure(d, full_fits, notable)

    # ---- write doc
    lt = lodo_tab.copy()
    a_all = lt.query("group=='all'").set_index("model")
    el = {g: full_fits[g][1] for g in GROUPS}
    cov_g = cover.groupby(d.group).mean()
    c_all = con[(con.group == "all")].set_index("RMSE(A) - RMSE(B)")
    naive_vs = c_all.loc["(a) naive full transfer - (c) shrunk: vmax + weight"]
    forty_vs = c_all.loc["(d) official forty + weight - (c) shrunk: vmax + weight"]
    rel_p = rel.set_index(["group", "target"])["Spearman-Brown reliability"].unstack()
    head = []
    for g in GROUPS + ("all",):
        L = lt[lt.group == g].set_index("model")
        C = con[con.group == g].set_index("RMSE(A) - RMSE(B)")
        nv = C.loc["(a) naive full transfer - (c) shrunk: vmax + weight"]
        fv = C.loc["(d) official forty + weight - (c) shrunk: vmax + weight"]
        head.append({"group": g, "n": int(L.n.iloc[0]),
                     "elasticity (c)": el.get(g, np.nan),
                     "RMSE naive": L.loc["(a) naive full transfer", "RMSE"],
                     "RMSE group mean": L.loc["(b) group mean", "RMSE"],
                     "RMSE weight only": L.loc["(b2) weight only", "RMSE"],
                     "RMSE shrunk": L.loc["(c) shrunk: vmax + weight", "RMSE"],
                     "naive - shrunk [95% CI]": f"{nv['yd/s']:.3f} [{nv['lo95']:.3f}, {nv['hi95']:.3f}]",
                     "RMSE forty+wt": L.loc["(d) official forty + weight", "RMSE"],
                     "forty - shrunk [95% CI]": f"{fv['yd/s']:+.3f} [{fv['lo95']:.3f}, {fv['hi95']:.3f}]",
                     "calib naive": L.loc["(a) naive full transfer", "calibration slope"],
                     "calib shrunk": L.loc["(c) shrunk: vmax + weight", "calibration slope"],
                     "80% PI coverage": cover[d.group == g].mean() if g != "all" else cover.mean()})
    head = pd.DataFrame(head)
    v = {"minp": MIN_PLAYS, "pi": np.nanmedian((pre.projHi80Mph - pre.projLo80Mph) / 2), "cov": cover.mean()}
    win_el = rob[rob.model == "elasticity (c), full sample"].set_index("group")["RMSE"]
    for g in GROUPS + ("all",):
        L = lt[lt.group == g].set_index("model")
        C = con[con.group == g].set_index("RMSE(A) - RMSE(B)")
        nv = C.loc["(a) naive full transfer - (c) shrunk: vmax + weight"]
        wv = C.loc["(b2) weight only - (c) shrunk: vmax + weight"]
        fv = C.loc["(d) official forty + weight - (c) shrunk: vmax + weight"]
        v.update({f"n_{g}": L.loc["(a) naive full transfer", "RMSE"], f"s_{g}": L.loc["(c) shrunk: vmax + weight", "RMSE"],
                  f"d_{g}": nv["yd/s"], f"dl_{g}": nv.lo95, f"dh_{g}": nv.hi95, f"pct_{g}": nv["% of A"],
                  f"w_{g}": wv["yd/s"], f"wl_{g}": wv.lo95, f"wh_{g}": wv.hi95,
                  f"f_{g}": fv["yd/s"], f"fl_{g}": fv.lo95, f"fh_{g}": fv.hi95,
                  f"cn_{g}": L.loc["(a) naive full transfer", "calibration slope"],
                  f"cs_{g}": L.loc["(c) shrunk: vmax + weight", "calibration slope"]})
        if g != "all":
            v.update({f"rq_{g}": rel_p.loc[g, TARGETS["q90"]], f"rp_{g}": rel_p.loc[g, TARGETS["p99"]],
                      f"rw_{g}": rel_p.loc[g, "window random intercept (context-adjusted)"], f"e_{g}": el[g],
                      f"ew_{g}": win_el[g], f"cov_{g}": cov_g[g]})
    summary = SUMMARY.format(v=v) + "\n" + md(head) + "\n"
    how = HOW.format(v=v)
    DOC.write_text("\n".join([
        "# Sunday Speed: projecting NFL game top speed from the Combine\n",
        f"_Generated {date.today().isoformat()} by `scripts/sunday_speed.py`. Per-player table: "
        "`data/processed/sunday_speed_players.parquet`._\n",
        summary, how,
        "## 1. Choosing the game-speed target (split-half by alternating games, Spearman-Brown)\n",
        f"Whole-play halves need >= {MIN_PLAYS // 2} plays per half. Window-intercept halves need >= {MIN_WIN // 2} "
        "windows per half and refit the context + window-length mixed model of `scripts/translation.py` on each half.\n",
        md(rel),
        "\nAcross players, the chosen target and the window intercept correlate r = "
        + ", ".join(f"{g} {agree[g]:.2f}" for g in GROUPS) + ". Top-N means of play peaks were rejected: they rise with "
        "snap count.\n",
        f"\nModeling sample (tracked 40, official forty, weight, >= {MIN_PLAYS} REG plays): "
        + ", ".join(f"{g} {int((d.group == g).sum())}" for g in GROUPS) + " players.\n",
        "## 2. Out-of-sample projection accuracy (yd/s)\n",
        "Models are fit on the log scale within group. (a) uses the training-group mean ratio of game to Combine vmax "
        "(elasticity 1); (b) is the training-group mean; (c0)-(e) are OLS on the log predictors. "
        "Calibration slope is the slope of observed on predicted (1 = ideal; < 1 = predictions too spread). "
        "It is undefined for the group mean.\n",
        "### Leave-one-draft-class-out (train on two classes, predict the third)\n", md(lodo_tab),
        "\n### Repeated 5-fold CV (40 repeats, mean)\n", md(kf_tab),
        "\n### Paired contrasts (LODO predictions, bootstrap over players)\n", md(con),
        "\n### Robustness: the context-adjusted route/rush-window level as the target (LODO, yd/s of window peak speed)\n",
        "The `elasticity` row reports the coefficient in the RMSE column.\n\n", md(rob),
        "\n![sunday speed](fig/sunday_speed.png)\n",
        "## 3. 2025 class: held-out projections vs actual (trained on 2023-24 only; first 8 picks per group)\n",
        md(c25_tab, 2),
        "\n## 4. Biggest misses of naive full transfer (LODO, all classes)\n",
        "**Tests fast, plays slower** (naive projection far above the game):\n", md(over, 2),
        "\n**Plays faster than he tests** (naive projection far below the game):\n", md(under, 2),
        "\n## 5. Payoff: illustrative separation value on vertical routes (WR)\n",
        f"**Rough illustration, not a validated effect.** Within a player, a 1-SD faster vertical rep "
        f"(SD = {vert['sd_within']:.2f} yd/s within player × route) comes with +{SEP_PER_SD:.2f} yd separation. "
        f"Across WRs, 1 yd/s of Sunday top speed goes with {k:.2f} yd/s of mean vertical-route peak speed. Chaining those gives "
        f"about {sep_per_yds:.3f} yd of separation per yd/s ({sep_per_yds / MPH:.3f} yd per mph) of projected Sunday "
        "speed, relative to the average WR. This borrows a within-player slope for between-player differences.\n",
        md(sep_tab, 3),
        "\n## 6. Pre-draft view: prospects with a tracked 40 but no usable NFL target (earliest picks)\n",
        "This shows how the tool is used before the draft: a projection and band from Combine data alone.\n",
        md(thin_tab, 2),
        f"\nAll {len(pre)} WR/DB/DL prospects with a tracked 40 are in the parquet "
        f"({int(pre.hasTarget.sum())} with an NFL target, {int((~pre.hasTarget).sum())} without).\n",
    ]))
    print(DOC.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
