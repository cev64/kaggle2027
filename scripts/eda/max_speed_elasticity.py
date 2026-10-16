"""Whole-play max-speed elasticity: per-player p99 of game speed (all REG frames, incl. after the
throw / catch / in pursuit) vs Combine vmax, within position group. Compared with the snap-to-throw
route elasticities in docs/translation.md, this tells whether game-speed compression is a usage effect
(capacity present but not used before the throw) or a capacity effect."""
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

cap = pd.read_parquet("data/processed/capacity_players.parquet")
cap = cap[cap.vmaxSource == "tracked"][["nflId", "combinePosition", "vmax", "combineWeight"]]
parts = []
for s in (2023, 2024, 2025):
    parts.append(pl.scan_parquet(f"data/processed/game_tracking_{s}_clean.parquet")
                 .filter(pl.col("seasonType").cast(pl.String) == "REG")
                 .group_by("nflId").agg(n=pl.len(), p99=pl.col("s").quantile(0.99), p999=pl.col("s").quantile(0.999),
                                        plays=pl.struct("gameId", "playId").n_unique()).collect())
g = pl.concat(parts).group_by("nflId").agg(pl.col("p99").mean(), pl.col("p999").mean(), pl.col("plays").sum()).to_pandas()
d = g.merge(cap, on="nflId")
d = d[d.plays >= 200]
rows = []
for grp in ("WR", "DB", "DL", "TE", "OL"):
    x = d[d.combinePosition == grp].copy()
    for q in ("p99", "p999"):
        for ctrl in ("", " + np.log(combineWeight)"):
            m = smf.ols(f"np.log({q}) ~ np.log(vmax){ctrl}", x).fit(cov_type="HC1")
            lo, hi = m.conf_int().loc["np.log(vmax)"]
            rows.append({"group": grp, "game speed": q, "weight ctrl": bool(ctrl), "n": len(x),
                         "elasticity": m.params["np.log(vmax)"], "lo95": lo, "hi95": hi,
                         "game/Combine median": (x[q] / x.vmax).median(), "r": np.corrcoef(np.log(x[q]), np.log(x.vmax))[0, 1]})
out = pd.DataFrame(rows)
print(out.round(3).to_markdown(index=False))
out.to_csv("data/processed/eda/max_speed_elasticity.csv", index=False)
