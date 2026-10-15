#!/usr/bin/env python
"""Speed Tax step 2: capacity usage on every regular-season WR/TE route and the player tax model.

    python scripts/speed_tax.py [--rebuild]

Builds data/processed/route_usage.parquet (one row per REG route with capacity fractions and
context), fits mixed models separating each receiver's habitual shortfall ("tax") from
situational demand, checks the tax's reliability, and writes docs/speed_tax.md plus
data/processed/speed_tax_players.parquet.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from bdb27 import tracking as T  # noqa: E402
from bdb27 import usage as U  # noqa: E402

PROC = ROOT / "data/processed"
RAW = ROOT / "data/raw"
ROUTES = PROC / "route_usage.parquet"
DOC = ROOT / "docs/speed_tax.md"
PP_COLS = ["gameId", "playId", "nflId", "routeRan", "receiverAlignment", "linedUpPosition", "teamCoverageManZone",
           "down", "yardsToGo", "passLength", "target", "separationAtPassForward", "inMotionAtBallSnap", "cushion",
           "passResult"]
SPEED_F = ("speedFrac ~ C(routeRan) + bs(tWindow, df=4) + C(align) + C(manZone) + room + motion + C(season)"
           " + cushionF + cushionMissing")
BURST_F = "burstFrac ~ C(routeRan) + C(align) + C(manZone) + cushionF + cushionMissing + C(season)"


def md(d: pd.DataFrame, digits: int = 3) -> str:
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    return d.to_markdown(index=False) + "\n"


def build_routes() -> pd.DataFrame:
    cr = T.load_table("combine_results", RAW)
    ids = cr.filter(pl.col("combinePosition").cast(pl.String).is_in(["WR", "TE"]))["nflId"].cast(pl.Int32)
    pp = T.load_table("player_play", RAW).select(PP_COLS).filter(pl.col("routeRan").is_not_null())
    pp = pp.with_columns(pl.col("nflId").cast(pl.Int32), pl.col("playId").cast(pl.Int32))
    parts = []
    for season in (2023, 2024, 2025):
        lf = (pl.scan_parquet(PROC / f"game_tracking_{season}_clean.parquet")
              .filter((pl.col("seasonType").cast(pl.String) == "REG") & pl.col("nflId").is_in(ids.implode()))
              .select("gameId", "playId", "nflId", "frameId", "framesFromSnap", "x", "y", "s", "event", "losX"))
        w = U.route_windows(lf).with_columns(season=pl.lit(season, pl.Int16))
        parts.append(w.join(pp, on=U.KEY, how="inner"))
        print(f"  {season}: {parts[-1].height} routes")
    routes = pl.concat(parts)
    cap = pl.read_parquet(PROC / "capacity_players.parquet").with_columns(pl.col("nflId").cast(pl.Int32))
    routes = U.attach_capacity(routes, cap).join(
        cap.select("nflId", pl.col("combinePosition").cast(pl.String), "displayName", "draftOverallPick",
                   pl.col("draftYear").alias("classYear")), on="nflId", how="left")
    routes.write_parquet(ROUTES)
    return routes.to_pandas()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["align"] = d["linedUpPosition"].fillna("WR").replace({"FB": "TE", "RB": "WR"})
    d["manZone"] = d["teamCoverageManZone"].fillna("UNK")
    d["room"] = (110 - d["losX"]).clip(0, 100)  # yards to the goal line
    d["motion"] = d["inMotionAtBallSnap"].fillna(False).astype(int)
    d["cushionMissing"] = d["cushion"].isna().astype(int)
    d["cushionF"] = d["cushion"].fillna(d["cushion"].median())
    d["half"] = (d["gameId"] % 2).astype(int)  # split by game, not by play
    return d


def fit_tax(d: pd.DataFrame, formula: str, label: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # lbfgs stalls at the variance boundary here; bfgs/powell/nm agree with each other
        m = smf.mixedlm(formula, d, groups=d["nflId"]).fit(method=["bfgs", "powell"], reml=True)
    re = pd.Series({k: v.iloc[0] for k, v in m.random_effects.items()}, name=label)
    var_u, var_e = float(m.cov_re.iloc[0, 0]), float(m.scale)
    return m, re, var_u, var_e


def eligible(d, col, min_n):
    n = d.groupby("nflId")[col].transform("count")
    return d[d[col].notna() & (n >= min_n)]


def reliability(d, formula, col, min_n=40):
    rows = {}
    halves = [fit_tax(eligible(d[d.half == h], col, min_n // 2), formula, f"h{h}")[1] for h in (0, 1)]
    hh = pd.concat(halves, axis=1).dropna()
    r = hh.corr().iloc[0, 1]
    rows["split-half r (by game)"] = r
    rows["Spearman-Brown"] = 2 * r / (1 + r)
    rows["players (split-half)"] = len(hh)
    seasons = {}
    for s in sorted(d.season.unique()):
        sd = eligible(d[d.season == s], col, min_n // 2)
        if sd.nflId.nunique() >= 10:
            seasons[s] = fit_tax(sd, formula, str(s))[1]
    pairs = []
    for a, b in ((2023, 2024), (2024, 2025)):
        if a in seasons and b in seasons:
            x = pd.concat([seasons[a], seasons[b]], axis=1).dropna()
            pairs.append((len(x), x.corr().iloc[0, 1]))
    if pairs:
        n = sum(p[0] for p in pairs)
        rows["year-over-year r (pooled pairs)"] = sum(p[0] * p[1] for p in pairs) / n
        rows["players (YoY pairs)"] = n
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    routes = build_routes() if args.rebuild or not ROUTES.exists() else pd.read_parquet(ROUTES)
    d = prepare(routes)

    cov = pd.DataFrame([{
        "REG routes (WR/TE)": len(d), "players": d.nflId.nunique(),
        "routes with capacity": int(d.speedFrac.notna().sum()), "players with capacity": d.loc[d.speedFrac.notna(), "nflId"].nunique(),
        "window ends at throw": (d.windowEnd == "pass").mean(), "sack": (d.windowEnd == "sack").mean(),
        "median window (s)": d.tWindow.median(), "routes reaching 10 yd of path": d.burst.notna().mean(),
    }])
    by_route = (d.dropna(subset=["speedFrac"]).groupby("routeRan")
                .agg(routes=("speedFrac", "size"), medWindow=("tWindow", "median"), medPeak=("peakSpeed", "median"),
                     speedFrac=("speedFrac", "median"), p90speedFrac=("speedFrac", lambda x: x.quantile(0.9)),
                     overCapacity=("speedFrac", lambda x: (x > 1).mean()), burstFrac=("burstFrac", "median"))
                .sort_values("speedFrac", ascending=False).reset_index())

    sp = eligible(d[d.windowEnd == "pass"], "speedFrac", 40)
    bu = eligible(d[(d.windowEnd == "pass") & (d.sAtSnap < 1.0) & (d.motion == 0)], "burstFrac", 40)
    m_s, tax_s, vu_s, ve_s = fit_tax(sp, SPEED_F, "speedTax")
    m_b, tax_b, vu_b, ve_b = fit_tax(bu, BURST_F, "burstTax")
    rel_s, rel_b = reliability(sp, SPEED_F, "speedFrac"), reliability(bu, BURST_F, "burstFrac")
    rel = pd.DataFrame({"speed tax": rel_s, "burst tax": rel_b}).reset_index().rename(columns={"index": "check"})
    var = pd.DataFrame([
        {"model": "speed used (peak / vmax)", "routes": len(sp), "players": sp.nflId.nunique(),
         "player SD": np.sqrt(vu_s), "residual SD": np.sqrt(ve_s), "ICC per route": vu_s / (vu_s + ve_s),
         "routes for reliability 0.7": 0.7 / 0.3 * ve_s / vu_s},
        {"model": "burst used (2-10 yd / Combine)", "routes": len(bu), "players": bu.nflId.nunique(),
         "player SD": np.sqrt(vu_b), "residual SD": np.sqrt(ve_b), "ICC per route": vu_b / (vu_b + ve_b),
         "routes for reliability 0.7": 0.7 / 0.3 * ve_b / vu_b},
    ])

    # player table: tax (BLUP, in capacity-fraction units), raw means, capacity, draft
    meta = d.groupby("nflId").agg(name=("displayName", "first"), pos=("combinePosition", "first"),
                                  classYear=("classYear", "first"), pick=("draftOverallPick", "first"),
                                  vmax=("vmax", "first"), burst2_10=("burst2_10", "first"),
                                  vmaxSource=("vmaxSource", "first"), routes=("speedFrac", "size"),
                                  meanSpeedFrac=("speedFrac", "mean"), meanSep=("separationAtPassForward", "mean"))
    players = meta.join(tax_s).join(tax_b).reset_index()
    players.to_parquet(PROC / "speed_tax_players.parquet")
    show = players.dropna(subset=["speedTax"]).sort_values("speedTax")
    cols = ["name", "pos", "classYear", "pick", "routes", "vmax", "meanSpeedFrac", "speedTax", "burstTax"]
    corr = players[["speedTax", "burstTax", "vmax", "burst2_10", "pick", "meanSep"]].corr(method="spearman").round(2)

    fe = m_s.params.filter(like="routeRan").sort_values()
    DOC.write_text("\n".join([
        "# Speed Tax: capacity usage on regular-season routes\n",
        f"_Generated {date.today().isoformat()} by `scripts/speed_tax.py`; window extraction in `src/bdb27/usage.py`._\n",
        "Each WR/TE route (REG, `route_ran` non-null) is measured from the snap to the throw: peak speed and burst "
        "(mean path speed from 2 to 10 yd). Both are expressed as fractions of the player's Combine capacity "
        "(`vmax`, `burst2_10`; see `docs/capacity.md`). A linear mixed model with a player random intercept "
        "separates situational demand from the player's habitual usage:\n",
        f"- speed: `{SPEED_F} + (1 | player)` (windows ending at a throw, players with at least 40 routes)\n"
        f"- burst: `{BURST_F} + (1 | player)` (no motion and no speed at the snap)\n",
        "The **tax** is the player's random intercept, in capacity-fraction units. Negative means he plays below "
        "the capacity his peers use in the same situations.\n",
        "## Coverage\n", md(cov),
        "\n## Capacity used by route (median per route)\n", md(by_route),
        "\n## Variance decomposition\n", md(var),
        "\n## Reliability of the tax (split by game parity; year-over-year from per-season fits)\n", md(rel),
        "\n## Route fixed effects in the speed model (relative to the reference route)\n",
        md(fe.rename("coef").reset_index().rename(columns={"index": "term"})),
        "\n## Player-level correlations (Spearman)\n", md(corr.reset_index().rename(columns={"index": ""}), 2),
        "\n## Caveats\n",
        "- **High reliability is expected, not a finding.** In-game speed is a stable player trait, and dividing it "
        "by a fixed Combine number keeps it stable. The tax is only interesting if it predicts something "
        "(within-player separation; step 3) or differs from absolute game speed.\n"
        "- **Regression to the mean.** Noise in the Combine denominator shows up as a negative tax–capacity "
        "correlation. `vmax` is precise (ICC 0.97), so the speed tax is little affected (ρ = −0.16). The Combine "
        "burst is noisier (ICC 0.82), and burst tax vs `burst2_10` has ρ = −0.42, so part of that is artefact. "
        "Fix in step 3 by shrinking the denominator or modelling log(game) ~ log(Combine) with a slope.\n"
        "- **Short windows.** Windows end at the throw (median 2.8 s), so deep routes are still accelerating. "
        "Speed used means capacity reached by the throw, which mixes acceleration and top speed.\n"
        "- Raw mean separation is confounded by route mix (deep routes separate less). Compare separation within "
        "player and route, not across players.\n",
        "\n## Lowest speed tax (plays furthest below capacity)\n", md(show[cols].head(10)),
        "\n## Highest speed tax\n", md(show[cols].tail(10).iloc[::-1]),
    ]))
    print(DOC.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
