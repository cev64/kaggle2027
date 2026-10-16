#!/usr/bin/env python
"""Why does Combine top speed transfer to games at only ~40 cents on the dollar?

    PYTHONPATH=src:scripts python scripts/shrinkage.py

Moderator analysis of the weight-controlled within-group elasticity of game speed on Combine vmax
(tracked 40 only) for WR, DB and DL. Two complementary designs:

* Window level (snap to throw/sack peak speed, ``data/processed/windows_<group>.parquet``):
  - player-level moderators (role, draft tier, usage, speed-curve shape): MixedLM with a player random
    intercept, ``logCapC × moderator`` plus ``weightC × moderator`` (so each stratum is weight-controlled);
  - play/season-level moderators (experience, quarter, score margin, alignment, route depth): OLS with player
    fixed effects and cluster-robust SEs, so the interaction is identified *within player*.
* Whole play (per-player p99 / p99.9 of all REG tracking frames): OLS of log quantile on log vmax ×
  moderator + log weight (HC1), and player-FE versions for experience and game state.

Also: measurement-error share of the shortfall, and a ceiling check (elasticity by speed quantile, share of
frames near capacity). Benjamini-Hochberg across all interaction tests.

Writes docs/shrinkage.md, docs/fig/shrinkage.png, data/processed/eda/shrinkage_tests.csv.
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
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from speed_tax import md  # noqa: E402
from translation import CTX, WINDOW, capacity_reliability, eligible, mixed, prepare  # noqa: E402

PROC, RAW = ROOT / "data/processed", ROOT / "data/raw"
DOC, FIG = ROOT / "docs/shrinkage.md", ROOT / "docs/fig/shrinkage.png"
OUT_CSV = PROC / "eda/shrinkage_tests.csv"
GROUPS = ("WR", "DB", "DL")
VERTICAL = ["GO", "POST", "CORNER", "WHEEL"]
SHORT = ["HITCH", "SLANT", "FLAT", "SCREEN", "ANGLE"]


# ============================================================================ moderators
def player_meta() -> pd.DataFrame:
    """One row per tracked-capacity WR/DB/DL: vmax, weight, draft tier, nfl_position, v10rel."""
    cap = pd.read_parquet(PROC / "capacity_players.parquet")
    cap = cap[(cap.vmaxSource == "tracked") & cap.combinePosition.isin(GROUPS)]
    pls = pd.read_csv(RAW / "players.csv", na_values="NA").rename(columns={"nfl_id": "nflId"})
    cap = cap.merge(pls[["nflId", "nfl_position", "draft_round"]], on="nflId", how="left")
    rnd = cap["draft_round"]
    cap["draftTier"] = np.where(rnd <= 2, "R1-2", np.where(rnd <= 4, "R3-4", "R5-7/UDFA"))
    cap["logW"] = np.log(cap["combineWeight"])
    cap["logVmax"] = np.log(cap["vmax"])
    return cap[["nflId", "combinePosition", "vmax", "logVmax", "combineWeight", "logW", "draftYear", "draftTier",
                "nfl_position", "v10rel", "tModel10"]]


def play_state() -> pl.DataFrame:
    """Per player-play (REG, not nullified): quarter, own-team score margin, team."""
    games = pl.read_csv(RAW / "games.csv", null_values="NA").filter(pl.col("season_type") == "REG").select(
        "game_id", "home_team_abbr")
    pp = (pl.scan_csv(RAW / "player_play.csv", null_values="NA")
          .select("game_id", "play_id", "nfl_id", "team_abbr", "quarter", "pre_snap_home_score",
                  "pre_snap_visitor_score", "route_ran", "play_nullified_by_penalty")
          .collect()
          .join(games, on="game_id", how="inner"))
    home = pl.col("team_abbr") == pl.col("home_team_abbr")
    diff = pl.col("pre_snap_home_score") - pl.col("pre_snap_visitor_score")
    return pp.with_columns(margin=pl.when(home).then(diff).otherwise(-diff)).select(
        pl.col("game_id").cast(pl.Int64).alias("gameId"), pl.col("play_id").cast(pl.Int32).alias("playId"),
        pl.col("nfl_id").cast(pl.Int32).alias("nflId"), "quarter", "margin", "route_ran",
        (pl.col("play_nullified_by_penalty") == "Y").alias("nullified"))


def usage(ps: pl.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """Snaps per game (routes per game for WR), REG games with >= 1 snap, per player."""
    grp = meta.set_index("nflId")["combinePosition"]
    x = ps.filter(~pl.col("nullified")).to_pandas()
    x["grp"] = x.nflId.map(grp)
    x = x[x.grp.notna() & ((x.grp != "WR") | x.route_ran.notna())]
    return x.groupby(["nflId", "gameId"]).size().groupby("nflId").mean().rename("snapsPerGame")


# ============================================================================ window data
def load_windows(group: str, meta: pd.DataFrame, ps: pl.DataFrame, spg: pd.Series) -> pd.DataFrame:
    d = prepare(pd.read_parquet(PROC / f"windows_{group}.parquet"))
    d = d[(d.vmaxSource == "tracked") & d.windowEnd.isin(["pass", "sack"]) & (d.peakSpeed > 0)]
    d = eligible(d, "peakSpeed", 40).copy()
    m = meta.set_index("nflId")
    d["weightC"] = d.nflId.map(m.combineWeight)
    d["weightC"] -= d["weightC"].mean()
    d["logGame"] = np.log(d.peakSpeed)
    d["logCapC"] = np.log(d.vmax) - np.log(d.vmax).mean()
    st = ps.select("gameId", "playId", "nflId", "quarter", "margin").to_pandas()
    d = d.merge(st, on=["gameId", "playId", "nflId"], how="left")
    # ---- player-level moderators
    maj = d.groupby("nflId")["align"].agg(lambda s: s.value_counts().idxmax())
    if group == "WR":
        d["role"] = d.nflId.map(maj).map({"WR": "outside", "SLOT_WR": "slot"})
    elif group == "DB":
        pos = d.nflId.map(m.nfl_position)
        d["role"] = np.select([pos.isin(["FS", "SS"]), d.nflId.map(maj) == "SLOT_CB", pos == "CB"],
                              ["safety", "slot CB", "outside CB"], None)
    else:
        d["role"] = d.nflId.map(maj).map({"EDGE": "edge", "INTERIOR_LINE": "interior"})
    d["draftTier"] = d.nflId.map(m.draftTier)
    players = pd.Series(d.nflId.unique())
    med = players.map(spg).median()
    d["usageTier"] = np.where(d.nflId.map(spg) >= med, "high usage", "low usage")
    v = players.map(m.v10rel)
    d["v10relZ"] = (d.nflId.map(m.v10rel) - v.mean()) / v.std()
    pv = np.log(players.map(spg))
    d["usageZ"] = (np.log(d.nflId.map(spg)) - pv.mean()) / pv.std()
    # ---- play / season-level moderators
    d["experience"] = d.season - d.nflId.map(m.draftYear) + 1
    d["expTier"] = np.where(d.experience >= 2, "year 2-3", "rookie")
    d["gameQ"] = np.where(d.quarter >= 4, "Q4/OT", "Q1-3")
    d["marginTier"] = np.where(d.margin.abs() >= 14, "margin >= 14", "within 13")
    if group == "WR":
        d["alignPlay"] = d["align"].map({"WR": "outside", "SLOT_WR": "slot"})
        d["depth"] = np.select([d.routeRan.isin(VERTICAL), d.routeRan.isin(SHORT)], ["vertical", "short"], "intermediate")
    elif group == "DB":
        d["alignPlay"] = d["align"].map({"CB": "outside CB", "SLOT_CB": "slot CB", "HIGH_SAFETY": "high safety"})
        d["depth"] = np.where(d.covAssign.str.startswith("DEEP"), "deep", "underneath")
    else:
        d["alignPlay"] = d["align"].map({"EDGE": "edge", "INTERIOR_LINE": "interior"})
        d["depth"] = None
    return d


# ============================================================================ model helpers
def _dummies(x: pd.DataFrame, mod: str, levels: list, ref: str, cap: str, wt: str | None):
    terms, caps = [], []
    for i, lev in enumerate(levels):
        if lev == ref:
            continue
        k = f"L{i}"
        x[f"m_{k}"] = (x[mod] == lev).astype(float)
        x[f"c_{k}"] = x[cap] * x[f"m_{k}"]
        terms += [f"m_{k}", f"c_{k}"]
        caps.append((lev, f"c_{k}"))
        if wt:
            x[f"w_{k}"] = x[wt] * x[f"m_{k}"]
            terms.append(f"w_{k}")
    return terms, caps


def _wald(params, cov, names):
    b = params[names].to_numpy()
    V = cov.loc[names, names].to_numpy()
    chi = float(b @ np.linalg.solve(V, b))
    return chi, len(names), float(stats.chi2.sf(chi, len(names)))


def _row(base, params, cov, names_slope, df_resid=None):
    est = sum(params[n] for n in names_slope)
    var = sum(cov.loc[a, b] for a in names_slope for b in names_slope)
    se = np.sqrt(var)
    return est, est - 1.96 * se, est + 1.96 * se


def window_moderator(group, d, mod, design, levels=None, ref=None, continuous=False):
    """Interaction of logCapC with a moderator on window peak speed. design = 'mixed' or 'fe'."""
    x = d[d[mod].notna()].copy()
    if continuous:
        x["c_z"] = x["logCapC"] * x[mod]
        x["w_z"] = x["weightC"] * x[mod]
        terms, caps = [mod, "c_z", "w_z"], [("per +1 SD", "c_z")]
        levels = []
    else:
        levels = levels or sorted(x[mod].unique())
        ref = ref or levels[0]
        x = x[x[mod].isin(levels)]
        terms, caps = _dummies(x, mod, levels, ref, "logCapC", "weightC" if design == "mixed" else None)
    rhs = " + ".join(terms)
    if design == "mixed":
        m, _ = mixed(f"logGame ~ logCapC + weightC + {rhs} + {CTX[group]}{WINDOW}", x, "m")
        params, cov = m.params, m.cov_params()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.ols(f"logGame ~ C(nflId) + {rhs} + {CTX[group]}{WINDOW}", x).fit(
                cov_type="cluster", cov_kwds={"groups": x["nflId"]})
        params, cov = m.params, m.cov_params()
    chi, df, p = _wald(params, cov, [c for _, c in caps])
    rows = []
    if design == "mixed" and not continuous:
        for lev in levels:
            names = ["logCapC"] + [c for l2, c in caps if l2 == lev]
            est, lo, hi = _row(None, params, cov, names)
            n_pl = x.loc[x[mod] == lev, "nflId"].nunique()
            rows.append(dict(level=lev, kind="stratum slope", est=est, lo95=lo, hi95=hi, players=n_pl,
                             windows=int((x[mod] == lev).sum())))
    for lev, c in caps:
        est, lo, hi = _row(None, params, cov, [c])
        rows.append(dict(level=f"{lev} - {ref}" if not continuous else lev, kind="difference", est=est, lo95=lo,
                         hi95=hi, players=x.loc[x[mod] == lev, "nflId"].nunique() if not continuous else x.nflId.nunique(),
                         windows=int((x[mod] == lev).sum()) if not continuous else len(x)))
    test = dict(group=group, data="windows", design=design, moderator=mod, chi2=chi, df=df, p=p,
                players=x.nflId.nunique(), n=len(x))
    for r in rows:
        r.update(group=group, data="windows", design=design, moderator=mod)
    return rows, test


# ============================================================================ whole-play frames
def frame_quantiles(meta: pd.DataFrame, ps: pl.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-player(-season[-state]) frame-speed quantiles over all REG frames, one season at a time."""
    ids = pl.Series(meta.nflId.astype("int32").to_numpy()).implode()
    vm = pl.from_pandas(meta[["nflId", "vmax"]].astype({"nflId": "int32"}))
    st = ps.filter(~pl.col("nullified")).select(
        "gameId", "playId", "nflId",
        q4=pl.col("quarter") >= 4, lop=pl.col("margin").abs() >= 14)
    qs = dict(p50=0.5, p90=0.9, p99=0.99, p999=0.999)
    aggs = [pl.len().alias("frames"), pl.struct("gameId", "playId").n_unique().alias("plays")] + [
        pl.col("s").quantile(q).alias(k) for k, q in qs.items()] + [
        (pl.col("s") >= 0.90 * pl.col("vmax")).mean().alias("share90"),
        (pl.col("s") >= 0.80 * pl.col("vmax")).mean().alias("share80")]
    out = {"season": [], "q4": [], "lop": []}
    for season in (2023, 2024, 2025):
        f = (pl.scan_parquet(PROC / f"game_tracking_{season}_clean.parquet")
             .filter((pl.col("seasonType").cast(pl.String) == "REG") & pl.col("nflId").is_in(ids))
             .select("gameId", "playId", "nflId", "s")
             .collect()
             .join(vm, on="nflId", how="inner"))
        out["season"].append(f.group_by("nflId").agg(aggs).with_columns(season=pl.lit(season)))
        f = f.join(st, on=["gameId", "playId", "nflId"], how="inner")
        for k in ("q4", "lop"):
            out[k].append(f.group_by("nflId", k).agg(aggs).with_columns(season=pl.lit(season)))
        del f
    return {k: pl.concat(v).to_pandas() for k, v in out.items()}


def pool_seasons(q: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Frame-weighted mean of log quantiles across seasons."""
    q = q.copy()
    cols = ["p50", "p90", "p99", "p999"]
    for c in cols:
        q[c] = np.log(q[c])
    for c in cols + ["share90", "share80"]:
        q[c] = q[c] * q["frames"]
    g = q.groupby(by)[cols + ["share90", "share80", "frames", "plays"]].sum()
    for c in cols + ["share90", "share80"]:
        g[c] = g[c] / g["frames"]
    return g.reset_index()  # p* now on log scale


def player_level(fq, meta, spg, maj_roles) -> pd.DataFrame:
    p = pool_seasons(fq["season"], ["nflId"]).merge(meta, on="nflId")
    p = p[p.plays >= 200].copy()
    p["snapsPerGame"] = p.nflId.map(spg)
    p["role"] = p.nflId.map(maj_roles)
    return p


def wholeplay_moderator(p, group, mod, q="p999", continuous=False, levels=None):
    x = p[(p.combinePosition == group) & p[mod].notna()].copy()
    x["lv"] = x.logVmax - x.logVmax.mean()
    x["lw"] = x.logW - x.logW.mean()
    if continuous:
        x["z"] = (x[mod] - x[mod].mean()) / x[mod].std()
        m = smf.ols(f"{q} ~ lv*z + lw*z", x).fit(cov_type="HC1")
        names, caps, ref = ["lv:z"], [("per +1 SD", "lv:z")], None
    else:
        levels = levels or sorted(x[mod].unique())
        ref = levels[0]
        x = x[x[mod].isin(levels)]
        terms, caps = _dummies(x, mod, levels, ref, "lv", "lw")
        m = smf.ols(f"{q} ~ lv + lw + {' + '.join(terms)}", x).fit(cov_type="HC1")
        names = [c for _, c in caps]
    params, cov = m.params, m.cov_params()
    chi, df, pv = _wald(params, cov, names)
    rows = []
    if not continuous:
        for lev in levels:
            nm = ["lv"] + [c for l2, c in caps if l2 == lev]
            est, lo, hi = _row(None, params, cov, nm)
            rows.append(dict(level=lev, kind="stratum slope", est=est, lo95=lo, hi95=hi,
                             players=int((x[mod] == lev).sum())))
    for lev, c in caps:
        est, lo, hi = _row(None, params, cov, [c])
        rows.append(dict(level=lev if continuous else f"{lev} - {ref}", kind="difference", est=est, lo95=lo, hi95=hi,
                         players=len(x)))
    for r in rows:
        r.update(group=group, data=f"whole play {q}", design="OLS HC1", moderator=mod)
    return rows, dict(group=group, data=f"whole play {q}", design="OLS HC1", moderator=mod, chi2=chi, df=df, p=pv,
                      players=len(x), n=len(x))


def wholeplay_fe(cells, meta, group, mod, q="p999", min_plays=60):
    """Player-FE: log quantile per player × cell ~ C(nflId) + cell + logVmax:cell (cluster by player)."""
    x = cells.merge(meta, on="nflId")
    x = x[(x.combinePosition == group) & (x.plays >= min_plays)].copy()
    x["lv"] = x.logVmax - x.logVmax.mean()
    x["m"] = x[mod].astype(float)
    x = x[x.groupby("nflId")["m"].transform("nunique") == 2]
    extra = " + C(season)" if "season" in x and mod == "later" else ""
    m = smf.ols(f"{q} ~ C(nflId) + m + lv:m{extra}", x).fit(cov_type="cluster", cov_kwds={"groups": x["nflId"]})
    est = m.params["lv:m"]
    lo, hi = m.conf_int().loc["lv:m"]
    p = m.pvalues["lv:m"]
    return (dict(group=group, data=f"whole play {q}", design="player FE", moderator=mod, level="difference",
                 kind="difference", est=est, lo95=lo, hi95=hi, players=x.nflId.nunique()),
            dict(group=group, data=f"whole play {q}", design="player FE", moderator=mod, chi2=(est / m.bse["lv:m"]) ** 2,
                 df=1, p=p, players=x.nflId.nunique(), n=len(x)))


# ============================================================================ ceiling / measurement error
def ceiling(p) -> pd.DataFrame:
    rows = []
    for g in GROUPS:
        x = p[p.combinePosition == g].copy()
        for q in ("p50", "p90", "p99", "p999"):
            m = smf.ols(f"{q} ~ logVmax + logW", x).fit(cov_type="HC1")
            lo, hi = m.conf_int().loc["logVmax"]
            rows.append(dict(group=g, quantile=q, elasticity=m.params["logVmax"], lo95=lo, hi95=hi,
                             **{"game/Combine median": np.exp(x[q] - x.logVmax).median()}, players=len(x)))
    return pd.DataFrame(rows)


def near_capacity(p) -> pd.DataFrame:
    rows = []
    for g in GROUPS:
        x = p[p.combinePosition == g].copy()
        x["vmaxTercile"] = pd.qcut(x.vmax, 3, labels=["slowest third", "middle", "fastest third"])
        for t, s in x.groupby("vmaxTercile", observed=True):
            rows.append({"group": g, "Combine vmax tercile": t, "players": len(s), "vmax (yd/s)": s.vmax.median(),
                         "p99.9 game (yd/s)": np.exp(s.p999).median(),
                         "p99.9 / vmax": np.exp(s.p999 - s.logVmax).median(),
                         "% frames >= 80% vmax": 100 * s.share80.median(),
                         "% frames >= 90% vmax": 100 * s.share90.median()})
    return pd.DataFrame(rows)


def measurement_error(base: pd.DataFrame) -> pd.DataFrame:
    rel = capacity_reliability()
    rel = rel[rel.capacity == "vmax"].set_index("group")
    rows = []
    for _, r in base.iterrows():
        lam = rel.loc[r.group, "reliability of player mean"]
        dis = r.est / lam
        rows.append({"group": r.group, "elasticity (weight-ctrl)": r.est, "vmax reliability": lam,
                     "disattenuated": dis, "share of shortfall (1 - b) due to Combine error": (dis - r.est) / (1 - r.est),
                     "reliability needed to explain all": r.est})
    return pd.DataFrame(rows)


# ============================================================================ main
def main() -> int:
    meta = player_meta()
    ps = play_state()
    spg = usage(ps, meta)
    tests, rows, base_rows, maj_roles = [], [], [], {}
    windows = {}
    for g in GROUPS:
        d = load_windows(g, meta, ps, spg)
        windows[g] = d
        maj_roles.update(d.groupby("nflId")["role"].first().to_dict())
        # baseline for reference
        m, _ = mixed(f"logGame ~ logCapC + weightC + {CTX[g]}{WINDOW}", d, "b")
        lo, hi = m.conf_int().loc["logCapC"]
        base_rows.append(dict(group=g, est=m.params["logCapC"], lo95=lo, hi95=hi, players=d.nflId.nunique(),
                              windows=len(d)))
        specs = [("role", "mixed", dict(levels={"WR": ["outside", "slot"], "DB": ["outside CB", "slot CB", "safety"],
                                                 "DL": ["edge", "interior"]}[g])),
                 ("draftTier", "mixed", dict(levels=["R1-2", "R3-4", "R5-7/UDFA"])),
                 ("usageTier", "mixed", dict(levels=["low usage", "high usage"])),
                 ("usageZ", "mixed", dict(continuous=True)),
                 ("v10relZ", "mixed", dict(continuous=True)),
                 ("expTier", "mixed", dict(levels=["rookie", "year 2-3"])),
                 ("expTier", "fe", dict(levels=["rookie", "year 2-3"])),
                 ("gameQ", "fe", dict(levels=["Q1-3", "Q4/OT"])),
                 ("marginTier", "fe", dict(levels=["within 13", "margin >= 14"])),
                 ("alignPlay", "fe", dict(levels={"WR": ["outside", "slot"], "DB": ["outside CB", "slot CB", "high safety"],
                                                  "DL": ["edge", "interior"]}[g]))]
        if g != "DL":
            specs.append(("depth", "fe", dict(levels={"WR": ["short", "intermediate", "vertical"],
                                                     "DB": ["underneath", "deep"]}[g])))
        for mod, design, kw in specs:
            r, t = window_moderator(g, d, mod, design, **kw)
            rows += r
            tests.append(t)
            print(f"  {g} {mod} {design}: p={t['p']:.3f}", flush=True)

    # ---------------- whole-play
    fq = frame_quantiles(meta, ps)
    p = player_level(fq, meta, spg, maj_roles)
    for g in GROUPS:
        x = p[p.combinePosition == g]
        p.loc[x.index, "usageTier"] = np.where(x.snapsPerGame >= x.snapsPerGame.median(), "high usage", "low usage")
    p["logSPG"] = np.log(p.snapsPerGame)
    for g in GROUPS:
        role_lv = {"WR": ["outside", "slot"], "DB": ["outside CB", "slot CB", "safety"], "DL": ["edge", "interior"]}[g]
        for q in ("p99", "p999"):
            for mod, kw in (("role", dict(levels=role_lv)), ("draftTier", dict(levels=["R1-2", "R3-4", "R5-7/UDFA"])),
                            ("usageTier", dict(levels=["low usage", "high usage"])), ("logSPG", dict(continuous=True)),
                            ("v10rel", dict(continuous=True))):
                r, t = wholeplay_moderator(p, g, mod, q, **kw)
                rows += r
                tests.append(t)
        # experience: player-season cells, within player
        ps_cells = fq["season"].copy()
        for c in ("p50", "p90", "p99", "p999"):
            ps_cells[c] = np.log(ps_cells[c])
        ps_cells = ps_cells.merge(meta[["nflId", "draftYear"]], on="nflId")
        ps_cells["later"] = (ps_cells.season - ps_cells.draftYear >= 1).astype(int)
        for q in ("p99", "p999"):
            r, t = wholeplay_fe(ps_cells.drop(columns="draftYear"), meta, g, "later", q)
            rows.append(r)
            tests.append(t)
            for k, mod in (("q4", "q4"), ("lop", "lop")):
                cells = pool_seasons(fq[k], ["nflId", k])
                cells[mod] = cells[k].astype(int)
                r, t = wholeplay_fe(cells, meta, g, mod, q)
                rows.append(r)
                tests.append(t)

    tests = pd.DataFrame(tests)
    tests["q_BH"] = multipletests(tests.p, method="fdr_bh")[1]
    res = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tests.to_csv(OUT_CSV, index=False)
    res.to_csv(OUT_CSV.with_name("shrinkage_estimates.csv"), index=False)
    base = pd.DataFrame(base_rows)
    ceil = ceiling(p)
    near = near_capacity(p)
    me = measurement_error(base)
    pd.to_pickle(dict(tests=tests, res=res, base=base, ceil=ceil, near=near, me=me, p=p),
                 PROC / "eda/shrinkage_results.pkl")
    print(md(tests))
    print(md(ceil))
    print(md(near))
    print(md(me))
    return 0


if __name__ == "__main__":
    sys.exit(main())
