"""Sprint capacity curves from Combine 40-yard-dash tracking.

Each attempt's speed trace is fit with the mono-exponential sprint model used in
field-based sprint profiling (Samozino et al. 2016; Morin et al. 2019)::

    v(t) = vmax * (1 - exp(-(t - t0) / tau))      for t >= t0, else 0

``t0`` is fitted, so clips that start a few frames late (the runner already
moving) or early are handled without trimming rules. From ``(vmax, tau)``:

* ``a0 = vmax / tau``: initial acceleration (yd/s^2), i.e. relative horizontal force F0/m
* ``d90``: distance to reach 90% of vmax = ``vmax * tau * (ln 10 - 0.9)``
* ``tModel10`` / ``tModel40``: curve-implied 10-yd split and 40 time from first movement
* ``pmaxRel``: peak relative horizontal power = ``a0 * vmax / 4`` (W/kg after unit conversion)

The model captures top speed well (retest ICC 0.97) but its acceleration constant is
poorly identified from a 3-point-stance start (tau ICC ~0.5). Burst is therefore also
measured empirically from interpolated split times on the run axis, with distance
anchored at first movement (the model's pre-clip distance is added): ``split2_10``
(2->10 yd), ``split10_20``, ``split20_40``, ``v10`` (speed at 10 yd), ``v10rel = v10/vmax``
and ``burst2_10 = 8 / split2_10`` (mean yd/s over 2-10 yd).

Speeds in yd/s; SI versions are added by :func:`add_si`. Air resistance is ignored
(about 3-5% of F0 at these speeds, near-constant across players).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

from .tracking import DT

YD = 0.9144  # m per yd
LB = 0.45359237  # kg per lb
SPLIT_MARKS = (2, 5, 10, 20, 40)
FIT_COLS = ["vmax", "tau", "t0", "a0", "d90", "tModel10", "tModel40", "pmaxRel", "rmse", "r2",
            "peakSpeedObs", "speedAt40Obs", "runLength", "nFrames", "dPreClip",
            "split2_10", "split10_20", "split20_40", "v10", "v10rel", "burst2_10"]


def sprint_speed(t, vmax, tau, t0):
    """Model speed at times ``t``."""
    dt = np.clip(np.asarray(t, float) - t0, 0, None)
    return vmax * (1 - np.exp(-dt / tau))


def sprint_distance(t, vmax, tau):
    """Model distance covered ``t`` seconds after first movement."""
    t = np.asarray(t, float)
    return vmax * (t - tau * (1 - np.exp(-t / tau)))


def time_to_distance(d, vmax, tau) -> float:
    """Seconds from first movement to cover ``d`` yards under the model."""
    hi = d / vmax + 5 * tau + 5
    return brentq(lambda t: sprint_distance(t, vmax, tau) - d, 0, hi)


def run_axis(x, y) -> np.ndarray:
    """Distance along the run's principal axis from the first frame (positive = direction of travel)."""
    p = np.c_[np.asarray(x, float) - x[0], np.asarray(y, float) - y[0]]
    u = np.linalg.svd(p - p.mean(0), full_matrices=False)[2][0]
    d = p @ u
    return d * (np.sign(d[-1]) or 1)


def fit_sprint(t, s) -> dict:
    """Fit (vmax, tau, t0) to one speed trace; returns parameters, derived capacity and fit quality."""
    t, s = np.asarray(t, float), np.asarray(s, float)
    p0 = (max(s.max(), 1.0) * 1.03, 1.0, -0.2)
    bounds = ([3.0, 0.2, -2.0], [14.0, 4.0, 1.5])
    (vmax, tau, t0), _ = curve_fit(sprint_speed, t, s, p0=p0, bounds=bounds, maxfev=20_000)
    resid = s - sprint_speed(t, vmax, tau, t0)
    ss_tot = ((s - s.mean()) ** 2).sum()
    a0 = vmax / tau
    return {"vmax": vmax, "tau": tau, "t0": t0, "a0": a0, "d90": vmax * tau * (np.log(10) - 0.9),
            "tModel10": time_to_distance(10, vmax, tau), "tModel40": time_to_distance(40, vmax, tau),
            "pmaxRel": a0 * vmax / 4, "rmse": float(np.sqrt((resid ** 2).mean())),
            "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan}


def attempt_profile(traj: pd.DataFrame, max_dist: float = 41.0) -> dict:
    """Capacity profile of one 40 attempt (player rows of one ``eventId``, any frame order).

    Uses reported speed ``s`` (matches position-derived speed, r = 0.999) for frames
    up to ``max_dist`` yd along the run axis.
    """
    g = traj.sort_values("frameId")
    t = (g["frameId"].to_numpy(float) - g["frameId"].iloc[0]) * DT
    d = run_axis(g["x"].to_numpy(), g["y"].to_numpy())
    s = g["s"].to_numpy(float)
    keep = d <= max_dist
    out = fit_sprint(t[keep], s[keep])
    out.update(empirical_splits(t, d, s, out["vmax"], out["tau"], out["t0"]))
    i40 = np.flatnonzero(d >= 40)
    out.update(peakSpeedObs=float(s[keep].max()), speedAt40Obs=float(s[i40[0]]) if len(i40) else np.nan,
               runLength=float(d.max()), nFrames=int(keep.sum()))
    return out


def empirical_splits(t, d, s, vmax, tau, t0) -> dict:
    """Split times between distance marks (from first movement) and speed at 10 yd.

    Distance covered before the clip starts (``t0 < 0``) is taken from the model and
    added, so marks are measured from first movement regardless of trimming. Marks the
    clip does not reach are NaN. Requires ``d`` to be (nearly) monotone, as in a sprint.
    """
    pre = float(sprint_distance(-t0, vmax, tau)) if t0 < 0 else 0.0
    dd = np.maximum.accumulate(d + pre)
    tm = {m: float(np.interp(m, dd, t)) if dd[0] <= m <= dd[-1] else np.nan for m in SPLIT_MARKS}
    v10 = float(np.interp(10, dd, s)) if dd[0] <= 10 <= dd[-1] else np.nan
    sp = tm[10] - tm[2]
    return {"dPreClip": pre, "split2_10": sp, "split10_20": tm[20] - tm[10], "split20_40": tm[40] - tm[20],
            "v10": v10, "v10rel": v10 / vmax, "burst2_10": 8 / sp if sp > 0 else np.nan}


def profile_attempts(df, key=("eventId",), meta=("nflId", "draftYear", "attempt")) -> pd.DataFrame:
    """:func:`attempt_profile` for every attempt in ``df`` (pandas or polars; ball rows dropped)."""
    df = df.to_pandas() if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame) else df
    if "isBall" in df.columns:
        df = df[~df["isBall"].astype(bool)]
    rows = []
    for k, g in df.groupby(list(key), sort=False):
        row = dict(zip(key, k if isinstance(k, tuple) else (k,)))
        row.update({c: g[c].iloc[0] for c in meta if c in g.columns})
        try:
            row.update(attempt_profile(g))
        except (RuntimeError, ValueError):
            row.update({c: np.nan for c in FIT_COLS})
        rows.append(row)
    return pd.DataFrame(rows)


def flag_fits(att: pd.DataFrame, min_r2: float = 0.97, max_rmse: float = 0.45, min_run: float = 38.0) -> pd.Series:
    """True for attempts usable as capacity estimates (good fit, full-length run)."""
    return (att["r2"] >= min_r2) & (att["rmse"] <= max_rmse) & (att["runLength"] >= min_run) & att["vmax"].notna()


PLAYER_PARAMS = ["vmax", "tau", "a0", "d90", "tModel10", "tModel40", "pmaxRel", "peakSpeedObs",
                 "split2_10", "split10_20", "split20_40", "v10", "v10rel", "burst2_10"]


def player_capacity(att: pd.DataFrame, params=PLAYER_PARAMS) -> pd.DataFrame:
    """Per-player mean of usable attempts plus the rep count; ``vmaxBest`` is the max over reps."""
    ok = att[att["usable"]] if "usable" in att.columns else att
    agg = ok.groupby("nflId")[list(params)].mean()
    agg["nReps"] = ok.groupby("nflId").size()
    agg["vmaxBest"] = ok.groupby("nflId")["vmax"].max()
    return agg.reset_index()


def add_si(cap: pd.DataFrame, weight_lb: str = "combineWeight") -> pd.DataFrame:
    """Add SI-unit capacity: vmax (m/s), F0 (N/kg and N), Pmax (W/kg and W) when weight is present."""
    out = cap.copy()
    out["vmaxMs"] = out["vmax"] * YD
    out["f0Rel"] = out["a0"] * YD  # N/kg
    out["pmaxRelW"] = out["pmaxRel"] * YD ** 2  # W/kg
    if weight_lb in out.columns:
        kg = out[weight_lb] * LB
        out["f0"] = out["f0Rel"] * kg
        out["pmax"] = out["pmaxRelW"] * kg
    return out
