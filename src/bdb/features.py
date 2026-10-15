"""Reusable spatial features on standardized tracking data (numpy / pandas).

Inputs are pandas (or polars, auto-converted) frames with ``gameId, playId,
frameId, nflId, x, y`` plus ``vx, vy`` (from :func:`bdb.tracking.add_kinematics`)
and a team column: ``side`` (offense/defense/ball) preferred, else ``club``.
Ball rows (``isBall``) are excluded from player-vs-player features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Voronoi

from .tracking import FIELD_LENGTH, FIELD_WIDTH

FRAME_KEY = ["gameId", "playId", "frameId"]
MAX_SPEED = 13.0  # yd/s, Fernandez & Bornn speed normalizer
RECEIVER_POSITIONS = {"WR", "TE", "RB", "FB"}


def _pd(df) -> pd.DataFrame:
    return df.to_pandas() if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame) else df


def _team_col(df: pd.DataFrame) -> str:
    for c in ("side", "club", "playerSide"):
        if c in df.columns:
            return c
    raise ValueError("Need a team column: 'side' (see tracking.add_side), 'club' or 'playerSide'")


def _players(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["isBall"].astype(bool)] if "isBall" in df.columns else df


# --------------------------------------------------------------- nearest players
def nearest_players(df, chunk: int = 20_000) -> pd.DataFrame:
    """Per player-frame: distance/id of nearest opponent and nearest teammate.

    Adds ``distNearestOpp``, ``nearestOppId``, ``distNearestMate``, ``nearestMateId``.
    Vectorized by padding each frame to a (frames, players) tensor.
    """
    df = _players(_pd(df)).sort_values(FRAME_KEY).reset_index(drop=True)
    team = df[_team_col(df)].astype(str).to_numpy()
    code = df.groupby(FRAME_KEY, sort=False).ngroup().to_numpy()
    slot = df.groupby(FRAME_KEY, sort=False).cumcount().to_numpy()
    F, P = code.max() + 1, slot.max() + 1
    xy = np.full((F, P, 2), np.nan)
    xy[code, slot] = df[["x", "y"]].to_numpy(float)
    tm = np.full((F, P), "", dtype=object)
    tm[code, slot] = team
    ids = np.full((F, P), np.nan)
    ids[code, slot] = df["nflId"].to_numpy(float)
    res = {k: np.full((F, P), np.nan) for k in ("dOpp", "iOpp", "dMate", "iMate")}
    eye = np.eye(P, dtype=bool)
    for s in range(0, F, chunk):
        sl = slice(s, s + chunk)
        d = np.linalg.norm(xy[sl, :, None, :] - xy[sl, None, :, :], axis=-1)
        same = tm[sl, :, None] == tm[sl, None, :]
        for key, mask in (("Opp", ~same), ("Mate", same & ~eye)):
            dm = np.where(mask & ~np.isnan(d), d, np.inf)
            j = dm.argmin(-1)
            best = np.take_along_axis(dm, j[..., None], -1)[..., 0]
            res["d" + key][sl] = np.where(np.isinf(best), np.nan, best)
            res["i" + key][sl] = np.where(np.isinf(best), np.nan, np.take_along_axis(ids[sl], j, -1))
    return df.assign(distNearestOpp=res["dOpp"][code, slot], nearestOppId=res["iOpp"][code, slot],
                     distNearestMate=res["dMate"][code, slot], nearestMateId=res["iMate"][code, slot])


def receiver_separation(df, positions=RECEIVER_POSITIONS) -> pd.DataFrame:
    """Separation (yd) of offensive pass-catchers from the nearest defender, per frame.

    Uses ``position``/``playerPosition`` if available to keep only ``positions``;
    otherwise every offensive player is returned.
    """
    df = _pd(df)
    if "distNearestOpp" not in df.columns:
        df = nearest_players(df)
    out = df[df[_team_col(df)].astype(str).str.lower() == "offense"] if "side" in df.columns else df
    pos = next((c for c in ("position", "playerPosition") if c in out.columns), None)
    if pos:
        out = out[out[pos].astype(str).isin(positions)]
    return out.assign(separation=out["distNearestOpp"])


def leverage(df) -> pd.DataFrame:
    """Leverage of each offensive player's nearest defender.

    ``leverageAngle`` (deg) is the defender's bearing from the receiver: 0 = straight
    downfield (+x, standardized), +90 = toward the field center (inside), -90 =
    toward the near sideline (outside), +/-180 = underneath. ``leverageSide`` is
    inside/outside/head_up (|lateral offset| <= 1 yd).
    """
    df = _pd(df)
    if "distNearestOpp" not in df.columns:
        df = nearest_players(df)
    off = df[df["side"].astype(str) == "offense"] if "side" in df.columns else df
    dpos = df[FRAME_KEY + ["nflId", "x", "y"]].rename(columns={"nflId": "nearestOppId", "x": "_dx", "y": "_dy"})
    m = off.merge(dpos, on=FRAME_KEY + ["nearestOppId"], how="left")
    inside = np.sign(FIELD_WIDTH / 2 - m["y"]).replace(0, 1)
    lat = (m["_dy"] - m["y"]) * inside
    m["leverageAngle"] = np.degrees(np.arctan2(lat, m["_dx"] - m["x"]))
    m["leverageSide"] = np.select([lat > 1, lat < -1], ["inside", "outside"], "head_up")
    return m.drop(columns=["_dx", "_dy"])


# -------------------------------------------------------------------- space
def voronoi_areas(frame, clip=True) -> pd.Series:
    """Voronoi cell area (sq yd) per player in one frame, bounded by the field.

    Uses the reflection trick: mirroring points across the four sidelines/end lines
    makes every original cell finite and exactly clipped to the field rectangle.
    """
    frame = _players(_pd(frame))
    pts = frame[["x", "y"]].to_numpy(float)
    if clip:
        pts = np.column_stack([pts[:, 0].clip(0, FIELD_LENGTH), pts[:, 1].clip(0, FIELD_WIDTH)])
    pts = pts + np.random.default_rng(0).normal(0, 1e-6, pts.shape)  # break exact ties
    n = len(pts)
    refl = [pts,
            np.c_[-pts[:, 0], pts[:, 1]], np.c_[2 * FIELD_LENGTH - pts[:, 0], pts[:, 1]],
            np.c_[pts[:, 0], -pts[:, 1]], np.c_[pts[:, 0], 2 * FIELD_WIDTH - pts[:, 1]]]
    vor = Voronoi(np.vstack(refl))
    areas = []
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        areas.append(ConvexHull(vor.vertices[region]).volume if region and -1 not in region else np.nan)
    return pd.Series(areas, index=frame.index, name="voronoiArea")


def add_voronoi(df) -> pd.DataFrame:
    """Add ``voronoiArea`` for every player-frame (loops over frames)."""
    df = _players(_pd(df))
    parts = [voronoi_areas(g) for _, g in df.groupby(FRAME_KEY, sort=False)]
    return df.assign(voronoiArea=pd.concat(parts))


def field_grid(step: float = 1.0, x_range=(0, FIELD_LENGTH), y_range=(0, FIELD_WIDTH)):
    """Meshgrid of cell centers covering the field."""
    xs = np.arange(x_range[0] + step / 2, x_range[1], step)
    ys = np.arange(y_range[0] + step / 2, y_range[1], step)
    return np.meshgrid(xs, ys)


def player_influence(pos, vel, ball, grid_xy) -> np.ndarray:
    """Fernandez & Bornn (2018) influence of each player on grid points.

    pos, vel: (P, 2); ball: (2,) or None; grid_xy: (G, 2). Returns (P, G) in [0, 1].
    Each player is a bivariate normal centered 0.5 s ahead, stretched along the
    direction of travel with speed and widened with distance to the ball (4-10 yd).
    """
    speed = np.linalg.norm(vel, axis=1).clip(0, MAX_SPEED - 0.01)
    srat = (speed / MAX_SPEED) ** 2
    if ball is None:
        radius = np.full(len(pos), 7.0)
    else:
        dist = np.linalg.norm(pos - np.asarray(ball, float), axis=1)
        radius = np.minimum(4 + dist ** 3 / (18 ** 3 / 6), 10)
    theta = np.arctan2(vel[:, 1], vel[:, 0])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)            # (P,2,2)
    sx, sy = radius * (1 + srat) / 2, np.maximum(radius * (1 - srat) / 2, 0.1)
    S2 = np.zeros((len(pos), 2, 2))
    S2[:, 0, 0], S2[:, 1, 1] = sx ** 2, sy ** 2
    cov = rot @ S2 @ np.transpose(rot, (0, 2, 1))
    inv = np.linalg.inv(cov)
    mu = pos + 0.5 * vel
    dz = grid_xy[None, :, :] - mu[:, None, :]                                   # (P,G,2)
    dp = (pos - mu)[:, None, :]
    q = np.einsum("pgi,pij,pgj->pg", dz, inv, dz)
    q0 = np.einsum("pgi,pij,pgj->pg", dp, inv, dp)
    return np.exp(-0.5 * (q - q0))


def pitch_control(frame, step: float = 1.0, ball_xy=None):
    """Offense control probability over a field grid for one frame.

    Returns ``(X, Y, control)`` where control = sigmoid(sum offense influence -
    sum defense influence). Ball position comes from the ball row, ``ball_xy``,
    or falls back to a constant influence radius.
    """
    frame = _pd(frame)
    if ball_xy is None and "isBall" in frame.columns and frame["isBall"].any():
        ball_xy = frame.loc[frame["isBall"].astype(bool), ["x", "y"]].iloc[0].to_numpy(float)
    pl_ = _players(frame)
    team = pl_[_team_col(pl_)].astype(str).str.lower().to_numpy()
    offense = team == "offense" if "offense" in team else team == team[0]
    X, Y = field_grid(step)
    grid = np.c_[X.ravel(), Y.ravel()]
    infl = player_influence(pl_[["x", "y"]].to_numpy(float), pl_[["vx", "vy"]].fillna(0).to_numpy(float), ball_xy, grid)
    ctrl = 1 / (1 + np.exp(-(infl[offense].sum(0) - infl[~offense].sum(0))))
    return X, Y, ctrl.reshape(X.shape)


# -------------------------------------------------------------------- pairs
def closing_speed(p1, v1, p2, v2) -> np.ndarray:
    """Rate (yd/s) at which two players close the distance between them (>0 = closing).

    Equals ``-d|p2 - p1|/dt``. Arrays broadcast over leading dims; last dim = (x, y).
    """
    rel, relv = np.asarray(p2, float) - np.asarray(p1, float), np.asarray(v2, float) - np.asarray(v1, float)
    dist = np.linalg.norm(rel, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -(rel * relv).sum(-1) / dist


def pair_closing(df, id1: int, id2: int) -> pd.DataFrame:
    """Per-frame distance and closing speed between players ``id1`` and ``id2`` in one play."""
    df = _pd(df)
    cols = ["frameId", "x", "y", "vx", "vy"]
    a = df[df["nflId"] == id1][cols]
    b = df[df["nflId"] == id2][cols]
    m = a.merge(b, on="frameId", suffixes=("1", "2"))
    p1, p2 = m[["x1", "y1"]].to_numpy(), m[["x2", "y2"]].to_numpy()
    return pd.DataFrame({"frameId": m["frameId"], "distance": np.linalg.norm(p2 - p1, axis=1),
                         "closingSpeed": closing_speed(p1, m[["vx1", "vy1"]].to_numpy(), p2, m[["vx2", "vy2"]].to_numpy())})
