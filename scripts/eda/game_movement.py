"""Game-movement mirror metrics: can the cohort's own REG-season tracking measure
the same kinematics as the Combine drills?

Five prototypes (see docs/eda/game_movement.md):
  1. Speed Tax      - in-game top speed / accel vs Combine 40 peak speed (all groups)
  2. Brake Check    - WR/TE route breaks (detect_cuts, snap..pass_forward) vs Combine route-drill cuts
                      and vs NGS separation_at_pass_forward
  3. Trigger Time   - DB response after pass_forward vs Combine back-pedal 45-degree transition
  4. DL first step  - post-snap burst vs NGS player_get_off, Combine 10-yd split, PASS_RUSH_DRILL burst
  5. OL pass set    - backward/lateral set speed in first 1 s on pass-block snaps vs Combine
                      PASS_PRO_MIRROR / wave drill and pressure allowed

Memory: each season is scanned lazily with column + row filters; peak RSS ~1.3 GB.
Play-level outputs are cached in data/processed/cache/gm_*.parquet (``--refresh`` rebuilds).

Run:  PYTHONPATH=src python scripts/eda/game_movement.py [--refresh]
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from bdb27.drills import detect_cuts, kinematics, response_after_event

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
PROC, RAW, CACHE = ROOT / "data/processed", ROOT / "data/raw", ROOT / "data/processed/cache"
CACHE.mkdir(parents=True, exist_ok=True)
SEASONS = (2023, 2024, 2025)
REFRESH = "--refresh" in sys.argv
BREAK_ROUTES = ["HITCH", "OUT", "IN", "SLANT", "POST", "CORNER"]  # no CURL/COMEBACK labels in route_ran
CUT_KW = dict(min_turn=25.0, min_drop=0.75)
RNG = np.random.default_rng(27)
RESULTS: dict = {}


# ------------------------------------------------------------------ stats helpers
def corr_ci(x, y, method="spearman"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 8:
        return {"r": np.nan, "lo": np.nan, "hi": np.nan, "n": n}
    r = stats.spearmanr(x, y)[0] if method == "spearman" else stats.pearsonr(x, y)[0]
    se = (np.sqrt(1.06 / (n - 3)) if method == "spearman" else 1 / np.sqrt(n - 3))
    z = np.arctanh(np.clip(r, -0.999, 0.999))
    return {"r": round(float(r), 3), "lo": round(float(np.tanh(z - 1.96 * se)), 3),
            "hi": round(float(np.tanh(z + 1.96 * se)), 3), "n": int(n)}


def fmt(c):
    return f"r={c['r']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] n={c['n']}"


def split_half(df, val, agg="mean", min_half=10, player="nflId", order=("gameId", "playId")):
    """Odd/even split of each player's plays (time order); Pearson r of half-means across
    players, Spearman-Brown corrected. Also the SB-projected reliability at the median
    full-sample play count (``rel_at_median_n``)."""
    d = df[[player, *order, val]].dropna().sort_values([player, *order])
    d["half"] = d.groupby(player).cumcount() % 2
    f = (lambda s: s.quantile(float(agg[1:]) / 100)) if isinstance(agg, str) and agg.startswith("q") else agg
    h = d.groupby([player, "half"])[val].agg(["count", f]).reset_index()
    h.columns = [player, "half", "n", "v"]
    if h["half"].nunique() < 2:
        return {"n_players": 0, "r_half": np.nan, "sb": np.nan}
    w = h.pivot(index=player, columns="half", values=["n", "v"]).dropna()
    w = w[(w[("n", 0)] >= min_half) & (w[("n", 1)] >= min_half)]
    if len(w) < 10:
        return {"n_players": len(w), "r_half": np.nan, "sb": np.nan}
    r = stats.pearsonr(w[("v", 0)], w[("v", 1)])[0]
    sb = 2 * r / (1 + r)
    n_half = float(np.median(w[("n", 0)]))
    n_full = float(d.groupby(player).size().median())
    k = n_full / n_half  # project single-half reliability r to median full n
    rel_med = k * r / (1 + (k - 1) * r)
    return {"n_players": int(len(w)), "median_plays_per_half": n_half, "r_half": round(float(r), 3),
            "sb": round(float(sb), 3), "median_plays_all": n_full, "rel_at_median_n": round(float(rel_med), 3)}


def player_mean(df, val, min_n=20, player="nflId", agg="mean"):
    g = df.dropna(subset=[val]).groupby(player)[val]
    out = g.quantile(float(agg[1:]) / 100) if agg.startswith("q") else g.agg(agg)
    return out[g.size() >= min_n]


def zscore_within(df, val, by):
    g = df.groupby(by)[val]
    return (df[val] - g.transform("mean")) / g.transform("std")


# ------------------------------------------------------------------ static tables
def load_static():
    cr = pd.read_csv(RAW / "combine_results.csv", na_values=["NA"])
    cr = cr.rename(columns={"nfl_id": "nflId"})
    for c in ("forty", "ten_yd_split"):
        cr[c] = pd.to_numeric(cr[c], errors="coerce")
    pp = (pl.scan_csv(RAW / "player_play.csv", infer_schema_length=20000, null_values=["NA", ""])
          .select(pl.col("game_id").alias("gameId"), pl.col("play_id").cast(pl.Int32).alias("playId"),
                  pl.col("nfl_id").cast(pl.Int32).alias("nflId"), "route_ran", "coverage_assignment",
                  "team_coverage_man_zone", "pass_result", "target",
                  *[pl.col(c).cast(pl.Float64, strict=False) for c in
                    ("separation_at_pass_forward", "player_get_off", "peak_pressure_probability_allowed",
                     "dropback_duration", "cushion")],
                  pl.col("pressure_allowed").cast(pl.String), pl.col("sack_allowed").cast(pl.String),
                  "lined_up_position")
          .collect())
    return cr, pp


# ------------------------------------------------------------------ per-season extraction
def season_scan(season):
    return (pl.scan_parquet(PROC / f"game_tracking_{season}_clean.parquet")
            .filter(pl.col("seasonType") == "REG"))


def extract_speed(season):
    """Per player-play max speed / accel (from 1 s pre-snap to end of tracking)."""
    lf = season_scan(season).filter(pl.col("framesFromSnap").is_not_null())
    return (lf.group_by("gameId", "playId", "nflId")
            .agg(nFrames=pl.len(), maxS=pl.col("s").max(), maxSs=pl.col("sSmooth").max(),
                 maxA=pl.col("a").max(), maxALong=pl.col("aLong").quantile(0.99),
                 lup=pl.col("linedUpPosition").first(), week=pl.col("week").first())
            .with_columns(season=pl.lit(season)).collect())


def route_break(g, xy, i_lo, i_hi, win=4, first_thresh=35.0, min_thresh=25.0, pick="first"):
    """Heading-based break detector (route_ran is known, so we only need *where* the break is).

    Turn signal dh[i] = heading(i+win) - heading(i-win) (0.8 s span). The break is the first
    local peak of |dh| >= ``first_thresh`` in [i_lo, i_hi] (``pick='max'``: the largest), else
    the largest if >= ``min_thresh``. The break point is the speed minimum within +/- win of it;
    entry = max speed in the 1 s before, re-acceleration = max dS/dt in the 1 s after.
    """
    k = kinematics(g, xy, 5)
    t, s, a, h, v = k["t"], k["speed"], k["dsdt"], k["heading"], k["vel"]
    n = len(s)
    lo, hi = max(i_lo, win), min(i_hi, n - 1 - win)
    if hi - lo < 2:
        return None
    idx = np.arange(lo, hi + 1)
    dh = np.abs(h[idx + win] - h[idx - win])
    dh[s[idx - win] < 1.5] = 0  # must be moving into the break
    peaks = [j for j in range(len(idx)) if dh[j] >= first_thresh and dh[j] == dh[max(0, j - win):j + win + 1].max()]
    if pick == "first" and peaks:
        j = peaks[0]
    else:
        j = int(np.argmax(dh))
        if dh[j] < min_thresh:
            return None
    c = idx[j]
    m = c - win + int(np.argmin(s[c - win:c + win + 1]))
    i0 = max(0, m - 10) + int(np.argmax(s[max(0, m - 10):m + 1]))
    i1 = m + int(np.argmax(s[m:min(n, m + 11)]))
    va, vb = v[max(0, c - win)], v[min(n - 1, c + win)]
    turn = np.degrees(np.arctan2(va[0] * vb[1] - va[1] * vb[0], np.dot(va, vb)))
    reg = np.flatnonzero(s[m:min(n, m + 16)] >= 0.9 * s[i0])
    dec_t = t[m] - t[i0]
    return {"tBreak": t[m], "absTurn": abs(turn), "entrySpeed": s[i0], "minSpeed": s[m], "exitSpeed": s[i1],
            "speedRetained": s[m] / s[i0] if s[i0] > 0 else np.nan, "decelTime": dec_t,
            "meanDecel": (s[i0] - s[m]) / dec_t if dec_t > 0 else 0.0,
            "peakDecel": max(0.0, -np.min(a[i0:m + 1])), "peakReaccel": np.max(a[m:min(n, m + 11)]),
            "timeToRegain": reg[0] * 0.1 if len(reg) else np.nan,
            "codTime": dec_t + (reg[0] * 0.1 if len(reg) else np.nan)}


def _break_row(g, pf_frame, snap_frame):
    f0 = g["frameId"].iloc[0]
    i_snap, i_pf = int(snap_frame - f0), int(pf_frame - f0)
    r = route_break(g, ("x", "y"), i_snap + 5, i_pf)
    if r is None:
        return None
    r["tCutFromSnap"] = r["tBreak"] - i_snap * 0.1
    cuts = detect_cuts(g, xy=("x", "y"), **CUT_KW)
    r["strictCut"] = bool(len(cuts) and ((cuts["tCut"] <= i_pf * 0.1 + 0.05) & (cuts["tCut"] >= i_snap * 0.1 + 0.5)).any())
    return r


def extract_breaks(season, ids, pp_routes):
    """WR/TE route breaks: frames from the snap to pass_forward + 1 s (extra second lets the
    re-acceleration peak exist), keeping the largest-drop cut whose minimum is <= pass_forward."""
    keys = pl.from_pandas(pp_routes[["gameId", "playId", "nflId", "route_ran"]]).with_columns(
        pl.col("gameId").cast(pl.Int64), pl.col("playId").cast(pl.Int32), pl.col("nflId").cast(pl.Int32))
    lf = (season_scan(season).filter(pl.col("nflId").is_in(ids))
          .select("gameId", "playId", "nflId", "frameId", "x", "y", "s", "event", "framesFromSnap", "snapFrame")
          .join(keys.lazy(), on=["gameId", "playId", "nflId"], how="inner"))
    df = lf.collect()
    pf = (df.filter(pl.col("event") == "pass_forward").group_by("gameId", "playId", "nflId")
          .agg(pf=pl.col("frameId").min()))
    df = (df.join(pf, on=["gameId", "playId", "nflId"], how="inner")
          .filter((pl.col("framesFromSnap") >= 0) & (pl.col("frameId") <= pl.col("pf") + 10)))
    n_frames = df.height
    rows = []
    for (gid, pid, nid), g in df.to_pandas().groupby(["gameId", "playId", "nflId"], sort=False):
        if len(g) < 15:
            continue
        r = _break_row(g, g["pf"].iloc[0], g["snapFrame"].iloc[0])
        base = {"gameId": gid, "playId": pid, "nflId": nid, "route_ran": g["route_ran"].iloc[0],
                "tPassFromSnap": (g["pf"].iloc[0] - g["snapFrame"].iloc[0]) * 0.1, "hasBreak": r is not None}
        rows.append({**base, **(r or {})})
    out = pd.DataFrame(rows)
    out["season"] = season
    return out, n_frames


def extract_trigger(season, ids, pp_cov):
    """DB reaction after pass_forward: response_after_event on frames snap .. pass_forward + 1.5 s."""
    keys = pl.from_pandas(pp_cov[["gameId", "playId", "nflId"]]).with_columns(
        pl.col("gameId").cast(pl.Int64), pl.col("playId").cast(pl.Int32), pl.col("nflId").cast(pl.Int32))
    df = (season_scan(season).filter(pl.col("nflId").is_in(ids))
          .select("gameId", "playId", "nflId", "frameId", "x", "y", "s", "event", "framesFromSnap", "xRelLos")
          .join(keys.lazy(), on=["gameId", "playId", "nflId"], how="inner").collect())
    pf = (df.filter(pl.col("event") == "pass_forward").group_by("gameId", "playId", "nflId")
          .agg(pf=pl.col("frameId").min()))
    df = (df.join(pf, on=["gameId", "playId", "nflId"], how="inner")
          .filter((pl.col("framesFromSnap") >= 0) & (pl.col("frameId") <= pl.col("pf") + 15)))
    n_frames = df.height
    rows = []
    for (gid, pid, nid), g in df.to_pandas().groupby(["gameId", "playId", "nflId"], sort=False):
        pff = g["pf"].iloc[0]
        if len(g) < 12 or g["frameId"].max() < pff + 10:
            continue
        r = response_after_event(g, int(pff), xy=("x", "y"), window_s=1.5, speed_gain=1.0)
        at = g[g["frameId"] == pff]
        rows.append({"gameId": gid, "playId": pid, "nflId": nid, **r,
                     "sAtPass": float(at["s"].iloc[0]) if len(at) else np.nan,
                     "depthAtPass": float(at["xRelLos"].iloc[0]) if len(at) else np.nan,
                     "absHeadingChange": abs(r["headingChange"]),
                     "speedGain1s": float(g.loc[g["frameId"] == pff + 10, "s"].iloc[0] - at["s"].iloc[0])
                     if len(at) and (g["frameId"] == pff + 10).any() else np.nan})
    out = pd.DataFrame(rows)
    out["season"] = season
    return out, n_frames


def extract_first_step(season, ids, frames_after=15):
    """Snap-relative displacement and speed for the first 1.5 s (DL burst, OL pass set)."""
    lf = (season_scan(season).filter(pl.col("nflId").is_in(ids))
          .filter(pl.col("framesFromSnap").is_between(0, frames_after))
          .select("gameId", "playId", "nflId", "framesFromSnap", "xSmooth", "ySmooth", "vxSmooth", "vySmooth",
                  "s", "xRelLos", "yRelCenter", "linedUpPosition"))
    df = lf.collect()
    n_frames = df.height
    k = ["gameId", "playId", "nflId"]
    df = df.sort(k + ["framesFromSnap"]).with_columns(
        dx=pl.col("xSmooth") - pl.col("xSmooth").first().over(k),
        dy=pl.col("ySmooth") - pl.col("ySmooth").first().over(k))
    df = df.with_columns(dist=(pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt(),
                         ySide=pl.col("yRelCenter").first().over(k).sign())

    def at(f, c):
        return pl.col(c).filter(pl.col("framesFromSnap") == f).first()

    first10 = pl.col("framesFromSnap") <= 10
    out = (
        df.lazy().group_by(k).agg(
            nF=pl.len(), lup=pl.col("linedUpPosition").first(),
            x0RelLos=at(0, "xRelLos"), y0=at(0, "yRelCenter"),
            d05=at(5, "dist"), d10=at(10, "dist"), s05=at(5, "s"), s10=at(10, "s"),
            dx05=at(5, "dx"), dx10=at(10, "dx"), dy10=at(10, "dy"),
            t3=(pl.col("framesFromSnap").filter(pl.col("s") >= 3.0).min() * 0.1),
            maxBackV=(-pl.col("vxSmooth")).filter(first10).max(),
            meanBackV=(-pl.col("vxSmooth")).filter(first10 & (pl.col("framesFromSnap") >= 1)).mean(),
            maxLatV=pl.col("vySmooth").abs().filter(first10).max(),
            maxS10=pl.col("s").filter(first10).max())
        .filter(pl.col("nF") >= 11).with_columns(season=pl.lit(season)).collect())
    return out.to_pandas(), n_frames


def cached(name, fn):
    p = CACHE / f"gm_{name}.parquet"
    if p.exists() and not REFRESH:
        return pd.read_parquet(p)
    t0 = time.time()
    df = fn()
    df.to_parquet(p, index=False)
    print(f"  built {name}: {len(df):,} rows in {time.time() - t0:.0f}s")
    return df


# ------------------------------------------------------------------ Combine-side metrics
def combine_metrics():
    ds = pd.read_parquet(PROC / "combine_drill_summary.parquet")
    forty = ds[ds["drillType"].str.contains("FORTY", na=False) | (ds["drillName"] == "FORTY_YARD_DASH")]
    c40 = forty.groupby("nflId").agg(c40Peak=("peakSpeed", "max"), c40PeakAccel=("peakAccel", "max"))

    ct = pl.scan_parquet(PROC / "combine_tracking_clean.parquet").filter(~pl.col("isBall"))
    canon = pl.col("drillCanon").cast(pl.String)

    def trajs(names):
        return (ct.filter(canon.is_in(names))
                .select("eventId", "nflId", "drillCanon", "frameId", "xd", "yd", "s").collect().to_pandas())

    # WR (and TE) route-drill breaks with the same detector as in game
    routes = ["CURL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT", "SPEED_OUT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT",
              "HOOK_ROUTE_RIGHT", "IN_ROUTE_LEFT"]
    rows = []
    for (eid, nid, dn), g in trajs(routes).groupby(["eventId", "nflId", "drillCanon"], observed=True):
        g = g.sort_values("frameId")
        mv = np.flatnonzero(g["s"].to_numpy() > 0.5)
        if not len(mv):
            continue
        r = route_break(g, ("xd", "yd"), int(mv[0]) + 5, len(g) - 1)
        if r is not None:
            rows.append({"nflId": nid, "drill": dn, "attempt": eid, **r})
    cb = pd.DataFrame(rows)

    # DB back-pedal 45-degree transition: biggest-turn cut (backpedal -> drive)
    rows = []
    for (eid, nid), g in trajs(["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"]).groupby(["eventId", "nflId"]):
        g = g.sort_values("frameId")
        r = route_break(g, ("xd", "yd"), 5, len(g) - 1, first_thresh=60, min_thresh=60, pick="max")
        if r is not None:
            rows.append({"nflId": nid, "eventId": eid, **r})
    bp = pd.DataFrame(rows)
    RESULTS["_n_bp_attempts"] = int(len(set(trajs(["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION"])["eventId"])))

    # DL PASS_RUSH_DRILL burst from motion onset; OL mirror / wave lateral quickness
    def burst(names):
        rows = []
        for (eid, nid), g in trajs(names).groupby(["eventId", "nflId"]):
            g = g.sort_values("frameId")
            s = g["s"].to_numpy()
            mv = np.flatnonzero(s > 0.5)
            if not len(mv):
                continue
            i0 = mv[0]
            pos = g[["xd", "yd"]].to_numpy()
            d = lambda k: float(np.linalg.norm(pos[min(i0 + k, len(pos) - 1)] - pos[i0]))
            hit3 = np.flatnonzero(s[i0:] >= 3.0)
            rows.append({"nflId": nid, "cd05": d(5), "cd10": d(10), "cs05": s[min(i0 + 5, len(s) - 1)],
                         "cs10": s[min(i0 + 10, len(s) - 1)], "ct3": hit3[0] * 0.1 if len(hit3) else np.nan})
        return pd.DataFrame(rows)

    pr = burst(["PASS_RUSH_DRILL"]).groupby("nflId").mean()
    ol = ds[ds["drillName"].isin(["PASS_PRO-MIRROR_DRILL", "PASS_PRO_MIRROR_DRILL",
                                  "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE"])].copy()
    ol["drill"] = np.where(ol["drillName"].str.contains("MIRROR"), "mirror", "wave")
    olw = ol.groupby(["nflId", "drill"])[["peakSpeed", "peakAccel", "peakDecel", "meanSpeedRetained"]].mean().unstack()
    olw.columns = [f"{d}_{m}" for m, d in olw.columns]
    return c40, cb, bp, pr, olw


# ------------------------------------------------------------------ main
def main():
    cr, pp = load_static()
    grp = cr.set_index("nflId")["combine_position"]
    ids = {g: cr.loc[cr["combine_position"].isin(g.split("/")), "nflId"].astype(int).tolist()
           for g in ("WR/TE", "DB", "DL", "OL")}
    ppd = pp.to_pandas()

    # ---------------- extraction (cached)
    nfr = {}

    def build_speed():
        return pd.concat([extract_speed(s).to_pandas() for s in SEASONS])

    sp = cached("speed", build_speed)

    def build_breaks():
        rr = ppd[ppd["route_ran"].isin(BREAK_ROUTES)]
        parts = []
        for s in SEASONS:
            o, n = extract_breaks(s, ids["WR/TE"], rr)
            nfr[f"breaks_{s}"] = n
            parts.append(o)
        return pd.concat(parts)

    br = cached("breaks", build_breaks)

    def build_trigger():
        cov = ppd[ppd["coverage_assignment"].notna() & (ppd["coverage_assignment"] != "NO_ASSIGNMENT")]
        parts = []
        for s in SEASONS:
            o, n = extract_trigger(s, ids["DB"], cov)
            nfr[f"trigger_{s}"] = n
            parts.append(o)
        return pd.concat(parts)

    tr = cached("trigger", build_trigger)

    def build_first(group):
        def f():
            parts = []
            for s in SEASONS:
                o, n = extract_first_step(s, ids[group])
                nfr[f"first_{group}_{s}"] = n
                parts.append(o)
            return pd.concat(parts)
        return f

    dl = cached("dl_first", build_first("DL"))
    ol = cached("ol_set", build_first("OL"))
    if nfr:
        old = json.load(open(CACHE / "gm_frames.json")) if (CACHE / "gm_frames.json").exists() else {}
        json.dump({**old, **nfr}, open(CACHE / "gm_frames.json", "w"))
    nfr = json.load(open(CACHE / "gm_frames.json")) if (CACHE / "gm_frames.json").exists() else {}
    RESULTS["frames_used"] = nfr

    c40, cb, bp, pr, olc = combine_metrics()

    for d in (sp, br, tr, dl, ol):
        d["gameId"] = d["gameId"].astype("int64")
        d["playId"] = d["playId"].astype("int32")
        d["nflId"] = d["nflId"].astype("int32")
    ppd["gameId"] = ppd["gameId"].astype("int64")
    ppd["playId"] = ppd["playId"].astype("int32")
    ppd["nflId"] = ppd["nflId"].astype("int32")
    K = ["gameId", "playId", "nflId"]

    # ================================================================ 1. SPEED TAX
    print("\n=== 1. SPEED TAX ===")
    sp["grp"] = sp["nflId"].map(grp)
    sp = sp[sp["maxS"] <= 12.5]  # drop teleport artefacts (> Combine max 11.85)
    RESULTS["speed"] = {"player_plays": len(sp), "players": int(sp["nflId"].nunique()),
                        "frames": int(sp["nFrames"].sum())}
    print(RESULTS["speed"])
    rel = {}
    for g in ["WR", "TE", "DB", "DL", "OL"]:
        d = sp[sp["grp"] == g]
        rel[g] = {m: split_half(d, v, agg=a, min_half=25) for m, v, a in
                  [("p95_maxS", "maxS", "q95"), ("p99_maxS", "maxS", "q99"), ("max_maxS", "maxS", "max"),
                   ("p95_maxA", "maxA", "q95")]}
    RESULTS["speed_rel"] = rel
    for g, v in rel.items():
        print(g, {m: (x["n_players"], x["sb"], x.get("rel_at_median_n")) for m, x in v.items()})
    pm = sp.groupby("nflId").agg(nPlays=("maxS", "size"), p95S=("maxS", lambda s: s.quantile(.95)),
                                 p99S=("maxS", lambda s: s.quantile(.99)), maxS=("maxS", "max"),
                                 p95A=("maxA", lambda s: s.quantile(.95)))
    pm = pm[pm["nPlays"] >= 50].join(c40, how="left").join(cr.set_index("nflId")[["combine_position", "forty"]])
    pm["share99"] = pm["p99S"] / pm["c40Peak"]
    pm["shareMax"] = pm["maxS"] / pm["c40Peak"]
    tab = []
    for g, d in pm.groupby("combine_position"):
        dd = d.dropna(subset=["c40Peak"])
        tab.append({"grp": g, "n_players": len(d), "n_with_40trk": len(dd),
                    "med_plays": int(d["nPlays"].median()),
                    "c40Peak_med": round(dd["c40Peak"].median(), 2), "game_p99_med": round(d["p99S"].median(), 2),
                    "game_max_med": round(d["maxS"].median(), 2),
                    "share_p99_med": round(dd["share99"].median(), 3), "share_max_med": round(dd["shareMax"].median(), 3),
                    "pct_exceed_40": round(float((dd["maxS"] > dd["c40Peak"]).mean()), 3),
                    "r_p99_c40": fmt(corr_ci(dd["p99S"], dd["c40Peak"])),
                    "r_max_c40": fmt(corr_ci(dd["maxS"], dd["c40Peak"])),
                    "r_p99_forty": fmt(corr_ci(d["p99S"], d["forty"]))})
    tab = pd.DataFrame(tab)
    print(tab.to_string())
    allc = pm.dropna(subset=["c40Peak"])
    RESULTS["speed_tab"] = tab.to_dict("records")
    RESULTS["speed_pooled"] = {"r_p99_c40": fmt(corr_ci(allc["p99S"], allc["c40Peak"])),
                               "r_p99_c40_withingrp": fmt(corr_ci(zscore_within(allc, "p99S", "combine_position"),
                                                                  zscore_within(allc, "c40Peak", "combine_position"))),
                               "r_p95A_c40Accel_withingrp": fmt(corr_ci(
                                   zscore_within(allc, "p95A", "combine_position"),
                                   zscore_within(allc, "c40PeakAccel", "combine_position")))}
    print(RESULTS["speed_pooled"])

    # ================================================================ 2. BRAKE CHECK
    print("\n=== 2. BRAKE CHECK ===")
    br = br.merge(ppd[K + ["separation_at_pass_forward", "target"]], on=K, how="left")
    br["grp"] = br["nflId"].map(grp)
    RESULTS["breaks_n"] = {"route_plays": len(br), "players": int(br["nflId"].nunique()),
                           "detect_rate": round(float(br["hasBreak"].mean()), 3),
                           "strict_detect_cuts_rate": round(float(br["strictCut"].fillna(False).astype(bool).mean()), 3),
                           "by_route": br.groupby("route_ran").agg(n=("hasBreak", "size"),
                                                                    detect=("hasBreak", "mean")).round(3).to_dict("index")}
    print(RESULTS["breaks_n"])
    b = br[br["hasBreak"]].copy()
    mets = ["entrySpeed", "minSpeed", "speedRetained", "meanDecel", "peakDecel", "peakReaccel", "absTurn",
            "decelTime", "codTime"]
    print(b.groupby("route_ran")[mets + ["tCutFromSnap"]].median().round(2).to_string())
    RESULTS["breaks_route_medians"] = b.groupby("route_ran")[mets + ["tCutFromSnap"]].median().round(2).to_dict("index")
    # route-adjust: z-score each metric within route type, then average per player
    for m in mets:
        b[m + "_z"] = zscore_within(b, m, "route_ran")
    relb = {m: split_half(b, m + "_z", min_half=15) for m in mets}
    RESULTS["breaks_rel"] = relb
    for m, v in relb.items():
        print(m, v)
    bp_ = b.groupby("nflId").agg(nBreaks=("entrySpeed", "size"), **{m: (m + "_z", "mean") for m in mets})
    bp_ = bp_[bp_["nBreaks"] >= 30]
    # hard (decelerating) breaks only: HITCH / OUT / IN
    bh = b[b["route_ran"].isin(["HITCH", "OUT", "IN"])]
    RESULTS["breaks_rel_hard"] = {m: split_half(bh, m + "_z", min_half=15) for m in mets}
    print("hard-break reliability:", {m: (v["n_players"], v["sb"], v.get("rel_at_median_n")) for m, v in RESULTS["breaks_rel_hard"].items()})
    bph = bh.groupby("nflId").agg(nBreaks=("entrySpeed", "size"), **{m: (m + "_z", "mean") for m in mets})
    bph = bph[bph["nBreaks"] >= 30]
    # Combine side: z within drill, average per player
    cbz = cb.copy()
    for m in mets[:-2] + ["decelTime", "codTime"]:
        cbz[m + "_z"] = zscore_within(cbz, m, "drill")
    cbz["grp"] = cbz["nflId"].map(grp)
    RESULTS["combine_cut_detect"] = cb.groupby("drill").size().to_dict()
    # combine test-retest across drills (odd/even drill split) for WR
    wr_c = cbz[cbz["drill"].isin(["CURL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT", "SPEED_OUT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT"])]
    cpl = wr_c.groupby("nflId")[[m + "_z" for m in mets]].mean()
    cpl["nDrills"] = wr_c.groupby("nflId").size()
    comb_rel = {}
    for m in mets:
        a = wr_c[wr_c["drill"].isin(["CURL_ROUTE_RIGHT", "COMEBACK_ROUTE_RIGHT"])].groupby("nflId")[m + "_z"].mean()
        c = wr_c[wr_c["drill"].isin(["SPEED_OUT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT"])].groupby("nflId")[m + "_z"].mean()
        j = pd.concat([a, c], axis=1).dropna()
        comb_rel[m] = fmt(corr_ci(j.iloc[:, 0], j.iloc[:, 1], "pearson"))
    RESULTS["combine_cut_crossdrill"] = comb_rel
    rr = wr_c.assign(k=wr_c.groupby(["nflId", "drill"]).cumcount())
    rr = rr[rr["k"] < 2].pivot_table(index=["nflId", "drill"], columns="k", values=[m + "_z" for m in mets]).dropna()
    RESULTS["combine_cut_samedrill_retest"] = {m: fmt(corr_ci(rr[(m + "_z", 0)], rr[(m + "_z", 1)], "pearson")) for m in mets}
    RESULTS["combine_route_breaks"] = {"attempts": len(cb), "players": int(cb["nflId"].nunique()),
                                       "by_drill_medians": cb.groupby("drill")[["entrySpeed", "minSpeed", "absTurn", "peakDecel"]].median().round(2).to_dict("index")}
    print("combine same-drill retest:", RESULTS["combine_cut_samedrill_retest"])
    print("combine cross-drill (curl+comeback vs speedout+dagger):", comb_rel)
    j = bp_.join(cpl, rsuffix="_c", how="inner")
    gc = {m: fmt(corr_ci(j[m], j[m + "_z"])) for m in mets}
    RESULTS["breaks_game_vs_combine"] = gc
    jh = bph.join(cpl, rsuffix="_c", how="inner")
    RESULTS["breaks_game_vs_combine_hard"] = {m: fmt(corr_ci(jh[m], jh[m + "_z"])) for m in mets}
    print("game(hard) vs combine (WR):", RESULTS["breaks_game_vs_combine_hard"])
    print("game vs combine (WR):", gc)
    # TE hook/in drills
    te_c = cbz[cbz["drill"].isin(["HOOK_ROUTE_RIGHT", "IN_ROUTE_LEFT"])].groupby("nflId")[[m + "_z" for m in mets]].mean()
    jt = bp_.join(te_c, how="inner")
    RESULTS["breaks_game_vs_combine_TE"] = {m: fmt(corr_ci(jt[m], jt[m + "_z"])) for m in ["entrySpeed", "speedRetained", "peakDecel", "peakReaccel"]}
    print("TE:", RESULTS["breaks_game_vs_combine_TE"])
    # separation: player level (route-adjusted separation) and play level within player
    sepd = br.dropna(subset=["separation_at_pass_forward"]).copy()
    sepd["sep_z"] = zscore_within(sepd, "separation_at_pass_forward", "route_ran")
    RESULTS["sep_rel"] = split_half(sepd, "sep_z", min_half=15)
    sep_p = sepd.groupby("nflId")["sep_z"].mean()
    j2 = bp_.join(sep_p)
    RESULTS["breaks_vs_sep_player"] = {m: fmt(corr_ci(j2[m], j2["sep_z"])) for m in mets}
    bs = b.dropna(subset=["separation_at_pass_forward"]).copy()
    bs["sep_z"] = zscore_within(bs, "separation_at_pass_forward", "route_ran")
    within = {}
    for m in mets:
        x = bs[m + "_z"] - bs.groupby("nflId")[m + "_z"].transform("mean")
        y = bs["sep_z"] - bs.groupby("nflId")["sep_z"].transform("mean")
        within[m] = fmt(corr_ci(x, y))
    RESULTS["breaks_vs_sep_play_within"] = within
    RESULTS["breaks_vs_sep_play_byroute"] = {
        r: {m: corr_ci(d[m], d["separation_at_pass_forward"])["r"] for m in ["speedRetained", "peakDecel", "entrySpeed", "absTurn"]}
        for r, d in bs.groupby("route_ran")}
    print("sep reliability:", RESULTS["sep_rel"])
    print("break vs sep player:", RESULTS["breaks_vs_sep_player"])
    print("break vs sep play (within player):", within)
    print("by route:", RESULTS["breaks_vs_sep_play_byroute"])
    jc = cpl.join(sep_p, how="inner")
    RESULTS["combine_cut_vs_sep"] = {m: fmt(corr_ci(jc[m + "_z"], jc["sep_z"])) for m in ["speedRetained", "peakDecel", "peakReaccel", "codTime"]}
    print("combine cut vs game sep:", RESULTS["combine_cut_vs_sep"])

    # ================================================================ 3. TRIGGER TIME
    print("\n=== 3. TRIGGER TIME ===")
    tr = tr.merge(ppd[K + ["coverage_assignment", "team_coverage_man_zone", "pass_result"]], on=K, how="left")
    tr["cov"] = np.where(tr["coverage_assignment"] == "MAN", "MAN",
                         np.where(tr["coverage_assignment"].str.startswith("DEEP") |
                                  tr["coverage_assignment"].str.startswith("QUARTERS") |
                                  tr["coverage_assignment"].str.contains("HALF", na=False), "DEEP_ZONE", "UNDER_ZONE"))
    RESULTS["trigger_n"] = {"player_plays": len(tr), "players": int(tr["nflId"].nunique()),
                            "by_cov": tr["cov"].value_counts().to_dict(),
                            "medians": tr[["tPeakAccel", "tSpeedGain", "absHeadingChange", "peakAccel", "speedGain1s", "sAtPass"]]
                            .median().round(3).to_dict(),
                            "share_tPeakAccel_at0": round(float((tr["tPeakAccel"] == 0).mean()), 3),
                            "share_no_gain": round(float(tr["tSpeedGain"].isna().mean()), 3)}
    print(RESULTS["trigger_n"])
    tmets = ["tPeakAccel", "tSpeedGain", "absHeadingChange", "peakAccel", "speedGain1s"]
    for m in tmets:
        tr[m + "_z"] = zscore_within(tr, m, "cov")
    RESULTS["trigger_rel"] = {m: split_half(tr, m + "_z", min_half=20) for m in tmets}
    RESULTS["trigger_rel_man"] = {m: split_half(tr[tr["cov"] == "MAN"], m, min_half=10) for m in tmets}
    for k_ in ("trigger_rel", "trigger_rel_man"):
        print(k_, {m: (v["n_players"], v["sb"], v.get("rel_at_median_n")) for m, v in RESULTS[k_].items()})
    # role-adjusted: residualize on coverage family, depth and speed at the throw
    X = pd.get_dummies(tr["cov"]).astype(float)
    X["depth"], X["sAt"], X["sAt2"] = tr["depthAtPass"], tr["sAtPass"], tr["sAtPass"] ** 2
    okr = X.notna().all(axis=1)
    for m in ["tPeakAccel", "speedGain1s", "absHeadingChange", "peakAccel"]:
        ok = okr & tr[m].notna()
        beta, *_ = np.linalg.lstsq(X[ok].to_numpy(), tr.loc[ok, m].to_numpy(), rcond=None)
        tr.loc[ok, m + "_res"] = tr.loc[ok, m] - X[ok].to_numpy() @ beta
    RESULTS["trigger_rel_resid"] = {m: split_half(tr, m + "_res", min_half=20)
                                    for m in ["tPeakAccel", "speedGain1s", "absHeadingChange", "peakAccel"]}
    print("resid:", {m: (v["n_players"], v["sb"], v.get("rel_at_median_n")) for m, v in RESULTS["trigger_rel_resid"].items()})
    tp = tr.groupby("nflId").agg(n=("tPeakAccel", "size"), **{m: (m + "_z", "mean") for m in tmets})
    tp = tp[tp["n"] >= 40]
    bpp = bp.groupby("nflId")[["codTime", "decelTime", "timeToRegain", "peakReaccel", "peakDecel", "minSpeed"]].mean()
    RESULTS["combine_bp45"] = {"attempts_with_transition": len(bp), "players": int(bp["nflId"].nunique()),
                               "attempts_total": RESULTS.pop("_n_bp_attempts")}
    bp_rel = bp.assign(k=bp.groupby("nflId").cumcount()).pivot(index="nflId", columns="k", values="codTime").dropna()
    RESULTS["combine_bp45_retest_codTime"] = fmt(corr_ci(bp_rel[0], bp_rel[1], "pearson")) if 1 in bp_rel else None
    jt = tp.join(bpp, how="inner")
    RESULTS["trigger_vs_bp45"] = {f"{g}~{c}": fmt(corr_ci(jt[g], jt[c])) for g in ["tPeakAccel", "tSpeedGain", "speedGain1s", "absHeadingChange"]
                                  for c in ["codTime", "peakReaccel", "timeToRegain"]}
    print(RESULTS["combine_bp45"], RESULTS["combine_bp45_retest_codTime"])
    print("trigger vs bp45:", RESULTS["trigger_vs_bp45"])

    # ================================================================ 4. DL FIRST STEP
    print("\n=== 4. DL FIRST STEP ===")
    dl = dl.merge(ppd[K + ["player_get_off"]], on=K, how="left")
    dl["dlpos"] = np.where(dl["lup"] == "EDGE", "EDGE", np.where(dl["lup"] == "INTERIOR_LINE", "IDL", "OTHER"))
    dg = dl[dl["player_get_off"].notna() & (dl["dlpos"] != "OTHER")].copy()  # pass-rush snaps with NGS get-off
    RESULTS["dl_n"] = {"snaps_all": len(dl), "snaps_with_getoff": len(dg), "players": int(dg["nflId"].nunique())}
    fmets = ["d05", "d10", "s05", "s10", "t3", "dx10"]
    RESULTS["dl_play_corr_getoff"] = {m: fmt(corr_ci(dg[m], dg["player_get_off"])) for m in fmets}
    print(RESULTS["dl_n"]); print("play-level vs get-off:", RESULTS["dl_play_corr_getoff"])
    for m in fmets + ["player_get_off"]:
        dg[m + "_z"] = zscore_within(dg, m, "dlpos")
    RESULTS["dl_rel"] = {m: split_half(dg, m + "_z", min_half=30) for m in fmets + ["player_get_off"]}
    print({m: (v["n_players"], v["sb"], v.get("rel_at_median_n")) for m, v in RESULTS["dl_rel"].items()})
    dpl = dg.groupby("nflId").agg(n=("d10", "size"), **{m: (m + "_z", "mean") for m in fmets + ["player_get_off"]})
    dpl = dpl[dpl["n"] >= 60]
    RESULTS["dl_player_corr_getoff"] = {m: fmt(corr_ci(dpl[m], dpl["player_get_off"])) for m in fmets}
    # within-player play-level: does get-off track the snap-relative burst after removing player means?
    RESULTS["dl_within_corr_getoff"] = {m: fmt(corr_ci(dg[m] - dg.groupby("nflId")[m].transform("mean"),
                                                       dg["player_get_off"] - dg.groupby("nflId")["player_get_off"].transform("mean")))
                                        for m in ["d10", "s10", "t3"]}
    print("player-level vs get-off:", RESULTS["dl_player_corr_getoff"])
    print("within-player:", RESULTS["dl_within_corr_getoff"])
    dc = dpl.join(cr.set_index("nflId")[["ten_yd_split", "forty"]]).join(pr, how="left").join(c40)
    RESULTS["dl_vs_combine"] = {f"{g}~{c}": fmt(corr_ci(dc[g], dc[c])) for g in ["d10", "s10", "t3", "player_get_off"]
                                for c in ["ten_yd_split", "cd10", "cs10", "ct3", "c40PeakAccel"]}
    print("DL vs combine:", RESULTS["dl_vs_combine"])

    # ================================================================ 5. OL PASS SET
    print("\n=== 5. OL PASS SET ===")
    ol = ol.merge(ppd[K + ["pressure_allowed", "peak_pressure_probability_allowed", "dropback_duration"]], on=K, how="left")
    og = ol[ol["pressure_allowed"].notna() & ol["lup"].isin(["T", "G", "C"])].copy()
    og["press"] = og["pressure_allowed"].astype(str).str.upper().eq("TRUE").astype(float)
    og["setDepth10"] = -og["dx10"]
    og["setWidth10"] = og["dy10"] * og["y0"].apply(np.sign).replace(0, 1)  # + = away from center
    og["setDist10"] = og["d10"]
    omets = ["maxBackV", "meanBackV", "maxLatV", "maxS10", "setDepth10", "setWidth10", "setDist10"]
    RESULTS["ol_n"] = {"pass_block_snaps": len(og), "players": int(og["nflId"].nunique()),
                       "by_pos_medians": og.groupby("lup")[omets].median().round(2).to_dict("index")}
    print(RESULTS["ol_n"])
    for m in omets + ["press", "peak_pressure_probability_allowed"]:
        og[m + "_z"] = zscore_within(og, m, "lup")
    RESULTS["ol_rel"] = {m: split_half(og, m + "_z", min_half=30) for m in omets + ["press", "peak_pressure_probability_allowed"]}
    print({m: (v["n_players"], v["sb"], v.get("rel_at_median_n")) for m, v in RESULTS["ol_rel"].items()})
    opl = og.groupby("nflId").agg(n=("maxBackV", "size"),
                                  **{m: (m + "_z", "mean") for m in omets + ["press", "peak_pressure_probability_allowed"]})
    opl = opl[opl["n"] >= 100]
    RESULTS["ol_vs_pressure_player"] = {m: fmt(corr_ci(opl[m], opl["peak_pressure_probability_allowed"])) for m in omets}
    RESULTS["ol_vs_pressrate_player"] = {m: fmt(corr_ci(opl[m], opl["press"])) for m in omets}
    RESULTS["ol_within_play"] = {m: fmt(corr_ci(og[m + "_z"] - og.groupby("nflId")[m + "_z"].transform("mean"),
                                                og["peak_pressure_probability_allowed"] - og.groupby("nflId")["peak_pressure_probability_allowed"].transform("mean")))
                                 for m in ["maxBackV", "setDist10"]}
    og["setDepth05"] = -og["dx05"]
    for m in ["maxBackV", "setDepth10", "setDepth05", "maxLatV"]:
        sub = og.dropna(subset=[m, "dropback_duration", "peak_pressure_probability_allowed"])
        xw = sub[m] - sub.groupby("nflId")[m].transform("mean")
        yw = sub["peak_pressure_probability_allowed"] - sub.groupby("nflId")["peak_pressure_probability_allowed"].transform("mean")
        dw = sub["dropback_duration"] - sub.groupby("nflId")["dropback_duration"].transform("mean")
        rx = xw - np.polyval(np.polyfit(dw, xw, 1), dw)
        ry = yw - np.polyval(np.polyfit(dw, yw, 1), dw)
        RESULTS.setdefault("ol_within_play_ctrl_dropback", {})[m] = fmt(corr_ci(rx, ry))
    RESULTS["ol_dropback_corr"] = {m: fmt(corr_ci(og[m], og["dropback_duration"])) for m in ["maxBackV", "setDepth10", "setDepth05"]}
    print("within ctrl dropback:", RESULTS["ol_within_play_ctrl_dropback"], RESULTS["ol_dropback_corr"])
    print("OL vs peak pressure prob:", RESULTS["ol_vs_pressure_player"])
    print("OL vs pressure rate:", RESULTS["ol_vs_pressrate_player"])
    print("within:", RESULTS["ol_within_play"])
    oc = opl.join(olc, how="left").join(c40)
    RESULTS["ol_vs_combine"] = {f"{g}~{c}": fmt(corr_ci(oc[g], oc[c])) for g in ["maxBackV", "maxLatV", "setDist10"]
                                for c in ["mirror_peakSpeed", "mirror_peakAccel", "mirror_peakDecel",
                                          "wave_peakSpeed", "wave_peakAccel", "c40PeakAccel"]}
    RESULTS["ol_combine_drill_vs_pressure"] = {c: fmt(corr_ci(oc[c], oc["peak_pressure_probability_allowed"]))
                                               for c in ["mirror_peakSpeed", "mirror_peakAccel", "wave_peakSpeed", "c40Peak"]}
    print("OL vs combine:", RESULTS["ol_vs_combine"])
    print("OL combine vs pressure:", RESULTS["ol_combine_drill_vs_pressure"])

    json.dump(RESULTS, open(CACHE / "gm_results.json", "w"), indent=1, default=str)
    import resource
    print("\nwrote", CACHE / "gm_results.json", "| peak RSS GB:", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))


if __name__ == "__main__":
    main()
