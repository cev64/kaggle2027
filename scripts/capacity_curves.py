#!/usr/bin/env python
"""Fit sprint capacity curves to every tracked Combine 40 and write the validation report.

    python scripts/capacity_curves.py

Inputs: data/processed/combine_tracking_clean.parquet, data/raw/combine_results.csv, players.csv.
Outputs: data/processed/capacity_attempts.parquet, data/processed/capacity_players.parquet,
docs/capacity.md and docs/fig/capacity_curves.png.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402

from bdb27 import capacity as K  # noqa: E402
from bdb27 import tracking as T  # noqa: E402

GROUPS = ["WR", "TE", "DB", "DL", "OL"]
OUT = ROOT / "data/processed"
DOC = ROOT / "docs/capacity.md"
FIG = ROOT / "docs/fig/capacity_curves.png"


def md(d: pd.DataFrame, digits: int = 3) -> str:
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    return d.to_markdown(index=False) + "\n"


def icc1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(1,1) for two measurements per subject."""
    x = np.c_[a, b]
    n, k = x.shape
    msb = k * ((x.mean(1) - x.mean()) ** 2).sum() / (n - 1)
    msw = ((x - x.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


def load():
    ct = (pl.scan_parquet(OUT / "combine_tracking_clean.parquet")
          .filter((pl.col("drillCanon") == "FORTY_YARD_DASH") & ~pl.col("isBall"))
          .select("eventId", "nflId", "draftYear", "attempt", "frameId", "x", "y", "s").collect())
    cr = T.load_table("combine_results", ROOT / "data/raw").to_pandas()
    pls = T.load_table("players", ROOT / "data/raw").to_pandas()
    return ct, cr, pls


def validation(att, cr):
    ok = att[att["usable"]].merge(cr[["nflId", "combinePosition", "forty", "tenYdSplit"]], on="nflId")
    # official time is the best rep; compare with each player's fastest curve-implied time
    best = ok.groupby(["nflId", "combinePosition"]).agg(tModel40=("tModel40", "min"), tModel10=("tModel10", "min"),
                                                         forty=("forty", "first"), ten=("tenYdSplit", "first")).reset_index()
    rows = []
    for g, d in [("all", best)] + [(g, best[best.combinePosition == g]) for g in GROUPS]:
        for model, off in (("tModel40", "forty"), ("tModel10", "ten")):
            x = d[[model, off]].dropna()
            if len(x) > 5:
                rows.append({"group": g, "time": "40" if off == "forty" else "10-yd split", "n": len(x),
                             "r": x[model].corr(x[off]), "bias_s": (x[model] - x[off]).mean(),
                             "mae_s": (x[model] - x[off]).abs().mean()})
    return pd.DataFrame(rows)


def reliability(att, params):
    ok = att[att["usable"]].sort_values(["nflId", "attempt"])
    two = ok.groupby("nflId").filter(lambda g: len(g) >= 2).groupby("nflId").head(2)
    a, b = two.groupby("nflId").nth(0).set_index("nflId"), two.groupby("nflId").nth(1).set_index("nflId")
    rows = []
    for p in params:
        x = pd.concat([a[p].rename("a"), b[p].rename("b")], axis=1).dropna()
        rows.append({"param": p, "nPlayers": len(x), "retest_r": x["a"].corr(x["b"]),
                     "ICC": icc1(x["a"].to_numpy(), x["b"].to_numpy()),
                     "within_sd": np.sqrt(((x["a"] - x["b"]) ** 2).mean() / 2)})
    return pd.DataFrame(rows)


def within_group_corr(cap):
    """Within-group Pearson r between the chosen capacities and the stopwatch."""
    pairs = [("vmax", "burst2_10"), ("vmax", "forty"), ("burst2_10", "forty"), ("burst2_10", "tenYdSplit"),
             ("burst2_10", "combineWeight"), ("vmax", "combineWeight")]
    rows = []
    for g in GROUPS:
        d = cap[cap.combinePosition == g]
        rows.append({"group": g, "n": int(d["vmax"].notna().sum()),
                     **{f"r({a},{b})": d[a].corr(d[b]) for a, b in pairs}})
    return pd.DataFrame(rows)


def redundancy(cap, params):
    """Share of each capacity parameter explained by stopwatch + size within group (5-fold CV R^2)."""
    rows = []
    for g in GROUPS:
        d = cap[(cap.combinePosition == g)].dropna(subset=["forty", "tenYdSplit", "combineWeight"])
        for p in params:
            x = d.dropna(subset=[p])
            if len(x) < 15:
                continue
            for name, cols in (("forty+weight", ["forty", "combineWeight"]),
                               ("forty+10yd+weight", ["forty", "tenYdSplit", "combineWeight"])):
                pred = cross_val_predict(LinearRegression(), x[cols], x[p], cv=5)
                rows.append({"group": g, "param": p, "controls": name, "n": len(x),
                             "cvR2": 1 - ((x[p] - pred) ** 2).sum() / ((x[p] - x[p].mean()) ** 2).sum()})
    return pd.DataFrame(rows).pivot_table(index=["group", "param", "n"], columns="controls", values="cvR2").reset_index()


def figure(cap):
    FIG.parent.mkdir(parents=True, exist_ok=True)
    colors = {"WR": "#2a6fdb", "DB": "#18a058", "TE": "#9b59b6", "DL": "#e67e22", "OL": "#7f8c8d"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    t = np.linspace(0, 5.5, 200)
    for g in GROUPS:
        d = cap[cap.combinePosition == g].dropna(subset=["vmax", "tau"])
        curves = np.array([K.sprint_speed(t, v, tau, 0) for v, tau in zip(d.vmax, d.tau)])
        dist = np.array([K.sprint_distance(t, v, tau) for v, tau in zip(d.vmax, d.tau)])
        grid = np.linspace(0, 40, 161)
        vd = np.array([np.interp(grid, dd, cc) for dd, cc in zip(dist, curves)])
        q = np.percentile(vd, [25, 50, 75], axis=0)
        ax[0].fill_between(grid, q[0], q[2], color=colors[g], alpha=0.18, lw=0)
        ax[0].plot(grid, q[1], color=colors[g], lw=2)
        nudge = {"WR": 0.18, "DB": -0.18}.get(g, 0)
        ax[0].text(40.5, q[1][-1] + nudge, g, color=colors[g], va="center", fontsize=10, fontweight="bold")
        ax[1].scatter(d.vmax, d.burst2_10, s=12, color=colors[g], alpha=0.65, label=g)
    ax[0].set(xlabel="Distance from first movement (yd)", ylabel="Speed (yd/s)", xlim=(0, 44),
              title="Combine 40 capacity curves (median, IQR band)")
    ax[1].set(xlabel="Top speed vmax (yd/s)", ylabel="Burst: mean speed 2–10 yd (yd/s)",
              title="Two capacities: burst vs top speed")
    ax[1].legend(frameon=False, ncol=5, fontsize=9, loc="upper left")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)


def main() -> int:
    ct, cr, pls = load()
    att = K.profile_attempts(ct)
    att["usable"] = K.flag_fits(att)
    att.to_parquet(OUT / "capacity_attempts.parquet")
    cap = player_cap = K.player_capacity(att)
    cap = cap.merge(cr[["nflId", "combinePosition", "combineWeight", "forty", "tenYdSplit", "ngsAthleticismScore"]],
                    on="nflId", how="left").merge(pls[["nflId", "displayName", "draftOverallPick", "draftYear"]],
                                                  on="nflId", how="left")
    cap = K.add_si(cap)
    cap.to_parquet(OUT / "capacity_players.parquet")

    params = ["vmax", "tau", "a0", "d90", "tModel10", "tModel40", "pmaxRel",
              "split2_10", "split10_20", "split20_40", "v10", "v10rel", "burst2_10"]
    fitq = pd.DataFrame([{"attempts": len(att), "fit failed": int(att["vmax"].isna().sum()),
                          "usable": int(att["usable"].sum()), "median R2": att["r2"].median(),
                          "median RMSE (yd/s)": att["rmse"].median(), "median t0 (s)": att["t0"].median(),
                          "players": att["nflId"].nunique(), "players usable": player_cap["nflId"].nunique()}])
    summ = (cap.groupby("combinePosition")[["vmax", "tau", "a0", "d90", "tModel40", "f0Rel", "pmaxRelW", "pmax"]]
            .median().reindex(GROUPS).reset_index())
    summ.insert(1, "n", cap.groupby("combinePosition").size().reindex(GROUPS).to_numpy())
    val, rel, red = validation(att, cr), reliability(att, params), redundancy(cap, ["vmax", "a0", "pmaxRel", "split2_10", "burst2_10", "v10rel", "split20_40"])
    corr = cap[["vmax", "a0", "d90", "split2_10", "v10rel", "split20_40"]].corr().round(2).reset_index().rename(columns={"index": ""})
    figure(cap)

    DOC.write_text("\n".join([
        "# Sprint capacity curves (Combine 40)\n",
        f"_Generated {date.today().isoformat()} by `scripts/capacity_curves.py`; model in `src/bdb27/capacity.py`._\n",
        "Each tracked 40 attempt's reported speed is fit with the mono-exponential sprint model "
        "v(t) = vmax(1 − e^{−(t−t0)/τ}) (Samozino et al. 2016). The start time t0 is a free parameter, which absorbs "
        "inconsistent clip trimming. Derived capacities: a0 = vmax/τ (initial acceleration = relative horizontal "
        "force F0/m), d90 (distance to 90% of vmax), curve-implied 10-yd and 40 times from first movement, and "
        "peak relative power a0·vmax/4. Usable attempts have R² ≥ 0.97, RMSE ≤ 0.45 yd/s and a run of at least 38 yd. "
        "Player values are means over usable reps.\n",
        "**Chosen capacities.**\n\n"
        "- **Top speed `vmax`**, from the model fit. Use the model, not the observed peak, because 97% of clips end "
        "while the runner is still accelerating or at speed.\n"
        "- **Burst `burst2_10`**: mean speed from 2 to 10 yd, timed from first movement. The model's τ and a0 are "
        "reported, but they are poorly identified from a 3-point-stance start.\n",
        "## Fit quality\n", md(fitq),
        "\n## Validation against the official stopwatch (fastest curve-implied time vs official best)\n", md(val),
        "\n## Retest reliability (players with 2 usable reps)\n", md(rel),
        "\n## Redundancy: CV R² of each capacity parameter from stopwatch + weight, within group\n",
        "Lower is better here: the remaining share is information the stopwatch does not carry.\n\n", md(red),
        "\n## Within-group correlations of the chosen capacities\n", md(within_group_corr(cap), 2),
        "\n## Parameter correlations (all players)\n", md(corr, 2),
        "\n## Group medians (yd/s, s, yd; SI: F0 N/kg, Pmax W/kg and W)\n", md(summ, 2),
        "\n![capacity curves](fig/capacity_curves.png)\n",
    ]))
    print(DOC.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
