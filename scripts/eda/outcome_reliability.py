"""Outcome-side EDA: sample sizes, over-expected models and reliability of NGS outcomes.

Regular season only. Run:  PYTHONPATH=src python scripts/eda/outcome_reliability.py

Writes
- data/processed/outcomes_player.parquet       per player x metric (n, raw mean, EB-shrunk mean)
- data/processed/outcomes_player_season.parquet per player x season x metric
- <stdout> + data/processed/cache/outcome_reliability_tables.md  markdown tables used in docs/eda/outcomes.md
"""
from __future__ import annotations

import os

# OpenMP oversubscription on a shared box makes HistGradientBoosting ~200x slower; one thread is fast enough.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "4")

import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT_MD = PROC / "cache" / "outcome_reliability_tables.md"
MIN_NS = (25, 50, 100, 200, 400)
RNG = np.random.default_rng(27)
_md: list[str] = []


def emit(s: str = "") -> None:
    print(s)
    _md.append(s)


def md_table(df: pl.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("")
            elif isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- data
def load() -> pl.DataFrame:
    pp = pl.read_csv(RAW / "player_play.csv", null_values="NA", infer_schema_length=50_000)
    games = pl.read_csv(RAW / "games.csv", null_values="NA").select("game_id", "season", "season_type", "week")
    cr = pl.read_csv(RAW / "combine_results.csv", null_values="NA").select("nfl_id", "combine_position")
    players = pl.read_csv(RAW / "players.csv", null_values="NA").select("nfl_id", "display_name", "draft_year")
    df = (
        pp.join(games, on="game_id")
        .filter(pl.col("season_type") == "REG")
        .join(cr, on="nfl_id")
        .join(players, on="nfl_id")
        .with_columns(
            pl.when(pl.col("pass_result") == "").then(None).otherwise(pl.col("pass_result")).alias("pass_result"),
            # yards to the opponent end zone
            pl.when(pl.col("yardline_side") == pl.col("possession_team"))
            .then(100 - pl.col("yardline_number"))
            .otherwise(pl.col("yardline_number"))
            .alias("yds_to_ez"),
            pl.col("in_motion_at_ball_snap").cast(pl.Int8),
            pl.col("game_clock").str.split(":").list.first().cast(pl.Int32, strict=False).alias("clock_min"),
        )
        .sort("nfl_id", "game_id", "play_id")
    )
    return df


# --------------------------------------------------------------------------- reliability
def anova_icc(ids: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """One-way random-effects ICC(1) with unequal group sizes (method of moments).

    Returns (icc, sigma2_between, sigma2_within)."""
    uniq, inv, n = np.unique(ids, return_inverse=True, return_counts=True)
    k, N = len(uniq), len(y)
    if k < 3:
        return np.nan, np.nan, np.nan
    means = np.bincount(inv, weights=y) / n
    grand = y.mean()
    ssb = np.sum(n * (means - grand) ** 2)
    ssw = np.sum((y - means[inv]) ** 2)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - np.sum(n**2) / N) / (k - 1)
    s2b = max((msb - msw) / n0, 0.0)
    return s2b / (s2b + msw), s2b, msw


def n_for(rel: float, icc: float) -> float:
    if not icc or icc <= 0 or np.isnan(icc):
        return np.inf
    return rel / (1 - rel) * (1 - icc) / icc


def split_half(d: pl.DataFrame, col: str, min_n: int, stat: str = "mean") -> tuple[int, float, float, float]:
    """Odd/even (within-player play order) split; returns (players, median n, r, Spearman-Brown)."""
    g = (
        d.filter(pl.col(col).is_not_null())
        .with_columns((pl.int_range(pl.len()).over("nfl_id") % 2).alias("_half"), pl.len().over("nfl_id").alias("_n"))
        .filter(pl.col("_n") >= min_n)
    )
    if g.height == 0:
        return 0, np.nan, np.nan, np.nan
    agg = pl.col(col).mean() if stat == "mean" else pl.col(col).std()
    h = g.group_by("nfl_id", "_half").agg(agg.alias("v"), pl.col("_n").first()).pivot(
        on="_half", index=["nfl_id", "_n"], values="v"
    )
    h = h.drop_nulls()
    if h.height < 8:
        return h.height, np.nan, np.nan, np.nan
    r = float(np.corrcoef(h["0"].to_numpy(), h["1"].to_numpy())[0, 1])
    sb = 2 * r / (1 + r) if r > -1 else np.nan
    return h.height, float(h["_n"].median()), r, sb


def yoy(d: pl.DataFrame, col: str, min_n: int = 50) -> tuple[int, float]:
    s = (
        d.filter(pl.col(col).is_not_null())
        .group_by("nfl_id", "season")
        .agg(pl.col(col).mean().alias("v"), pl.len().alias("n"))
        .filter(pl.col("n") >= min_n)
    )
    nxt = s.with_columns((pl.col("season") - 1).alias("season"))
    pairs = s.join(nxt, on=["nfl_id", "season"], suffix="_next")
    if pairs.height < 8:
        return pairs.height, np.nan
    return pairs.height, float(np.corrcoef(pairs["v"].to_numpy(), pairs["v_next"].to_numpy())[0, 1])


def reliability_row(d: pl.DataFrame, group: str, col: str, label: str, stat: str = "mean") -> dict:
    x = d.filter(pl.col(col).is_not_null())
    row: dict = {"group": group, "metric": label}
    if stat == "mean":
        icc, s2b, s2w = anova_icc(x["nfl_id"].to_numpy(), x[col].to_numpy().astype(float))
        row.update(ICC=icc, n_rel70=n_for(0.7, icc), k_shrink=(s2w / s2b if s2b > 0 else np.inf))
    else:
        row.update(ICC=np.nan, n_rel70=np.nan, k_shrink=np.nan)
    for m in (50, 100, 200):
        p, _, _, sb = split_half(x, col, m, stat)
        row[f"SB@{m}"] = sb
        row[f"pl@{m}"] = p
    if stat == "mean":
        np_, r = yoy(x, col, 50)
        row["yoy_pairs"] = np_
        row["yoy_r"] = r
    return row


# --------------------------------------------------------------------------- OE models
def _prep_X(d: pl.DataFrame, cats: list[str], nums: list[str]) -> tuple[np.ndarray, list[bool]]:
    cols, is_cat = [], []
    for c in cats:
        codes = d[c].cast(pl.String).cast(pl.Categorical).to_physical().cast(pl.Float64).to_numpy()
        cols.append(codes)
        is_cat.append(True)
    for c in nums:
        if d[c].drop_nulls().n_unique() < 2:  # e.g. in_motion_at_ball_snap is null on defensive rows
            continue
        cols.append(d[c].cast(pl.Float64).to_numpy())
        is_cat.append(False)
    return np.column_stack(cols), is_cat


def fit_oe(d: pl.DataFrame, y: str, cats: list[str], nums: list[str], binary: bool = False) -> tuple[np.ndarray, dict]:
    """5-fold GroupKFold (by player) out-of-fold expected values. Also an additive baseline (cats one-hot + nums)."""
    X, is_cat = _prep_X(d, cats, nums)
    yy = d[y].cast(pl.Float64).to_numpy()
    groups = d["nfl_id"].to_numpy()
    oof = np.zeros_like(yy)
    oof_lin = np.zeros_like(yy)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, yy, groups):
        if binary:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                               categorical_features=is_cat, random_state=0)
            m.fit(X[tr], yy[tr].astype(int))
            oof[te] = m.predict_proba(X[te])[:, 1]
        else:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                              categorical_features=is_cat, random_state=0)
            m.fit(X[tr], yy[tr])
            oof[te] = m.predict(X[te])
        # additive baseline: depth-1 boosting == additive main effects, no interactions
        lin = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_depth=1, min_samples_leaf=100,
                                            categorical_features=is_cat, random_state=0)
        lin.fit(X[tr], yy[tr])
        oof_lin[te] = lin.predict(X[te])
    info = {"target": y, "n": len(yy), "R2_additive": r2_score(yy, oof_lin), "R2_gbm": r2_score(yy, oof)}
    if binary:
        info["AUC"] = roc_auc_score(yy.astype(int), oof)
        info["Brier_skill"] = 1 - np.mean((yy - oof) ** 2) / np.var(yy)
    return oof, info


# --------------------------------------------------------------------------- DB targeted-defender proxy
_SUFFIX = re.compile(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)$")


def abbr(name: str) -> str:
    name = _SUFFIX.sub("", name.strip())
    parts = name.split(" ")
    return f"{parts[0][0]}.{' '.join(parts[1:])}"


def db_named_flags(db: pl.DataFrame) -> pl.DataFrame:
    """Flag DB player-plays where the DB is credited in the description of a targeted pass.

    `named_paren`: the DB's 'F.Last' appears inside (...) (tackler on C, pass defender on I).
    `named_int`  : 'INTERCEPTED by F.Last'.
    """
    db = db.with_columns(pl.col("display_name").map_elements(abbr, return_dtype=pl.String).alias("_abbr"))
    desc = db["play_description"].fill_null("").to_list()
    ab = db["_abbr"].to_list()
    paren, inter = [], []
    for s, a in zip(desc, ab):
        s1 = s.split("PENALTY")[0]
        groups = re.findall(r"\(([^()]*)\)", s1)
        names = ";".join(groups)
        paren.append(bool(re.search(r"(^|[;\s])" + re.escape(a) + r"(?=[;\s.,]|$)", names)))
        inter.append(("INTERCEPTED by " + a) in s1)
    return db.with_columns(pl.Series("named_paren", paren), pl.Series("named_int", inter))


# --------------------------------------------------------------------------- tracking speed
def tracking_speed() -> pl.DataFrame:
    """Per player-play max speed after the snap (REG only), streamed per season file."""
    out = []
    for yr in (2023, 2024, 2025):
        lf = (
            pl.scan_parquet(PROC / f"game_tracking_{yr}_clean.parquet")
            .filter((pl.col("seasonType") == "REG") & (~pl.col("isBall")) & (pl.col("framesFromSnap") >= 0))
            .group_by("gameId", "playId", "nflId")
            .agg(pl.col("s").max().alias("max_speed"), pl.col("s").quantile(0.9).alias("p90_speed"))
        )
        out.append(lf.collect(engine="streaming"))
    t = pl.concat(out).rename({"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id"})
    return t.with_columns(pl.col("play_id").cast(pl.Int64), pl.col("nfl_id").cast(pl.Int64))


# --------------------------------------------------------------------------- main
def sample_sizes(d: pl.DataFrame, group: str, col: str, label: str) -> tuple[dict, list[dict]]:
    x = d.filter(pl.col(col).is_not_null())
    per = x.group_by("nfl_id").len()
    n = per["len"].to_numpy()
    row = {"group": group, "outcome": label, "plays": x.height, "players": len(n),
           "p25": int(np.percentile(n, 25)) if len(n) else 0, "median": int(np.median(n)) if len(n) else 0,
           "p75": int(np.percentile(n, 75)) if len(n) else 0, "max": int(n.max()) if len(n) else 0,
           ">=50": int((n >= 50).sum()), ">=100": int((n >= 100).sum()), ">=200": int((n >= 200).sum())}
    ps = x.group_by("nfl_id", "season").len()
    srows = []
    for s in (2023, 2024, 2025):
        m = ps.filter(pl.col("season") == s)["len"].to_numpy()
        srows.append({"group": group, "outcome": label, "season": s, "players": len(m),
                      "median": int(np.median(m)) if len(m) else 0, ">=50": int((m >= 50).sum()),
                      ">=100": int((m >= 100).sum()), ">=200": int((m >= 200).sum())})
    return row, srows


def main() -> None:
    df = load()
    emit(f"REG player-plays (cohort): {df.height:,}")

    metrics: list[tuple[str, pl.DataFrame, str, str]] = []  # (group, frame, col, label)
    oe_info: list[dict] = []

    # ---------------- WR / TE: separation
    wr = df.filter(pl.col("combine_position").is_in(["WR", "TE"]) & pl.col("route_ran").is_not_null())
    miss = (
        wr.group_by(pl.col("pass_result").fill_null("run/none").alias("pass_result"))
        .agg(pl.len().alias("routes"), pl.col("separation_at_pass_forward").is_not_null().mean().alias("sep_recorded"))
        .sort("routes", descending=True)
    )
    tgt = wr.group_by("target").agg(pl.len().alias("routes"),
                                    pl.col("separation_at_pass_forward").is_not_null().mean().alias("sep_recorded"),
                                    pl.col("separation_at_pass_forward").mean().alias("mean_sep"))
    emit("\n### Separation recorded, by pass_result (WR/TE routes)\n")
    emit(md_table(miss))
    emit("\n### Separation by target\n")
    emit(md_table(tgt))

    sep = wr.filter(pl.col("separation_at_pass_forward").is_not_null())
    sep_cats = ["route_ran", "receiver_alignment", "lined_up_position", "team_coverage_man_zone", "team_coverage_type",
                "offense_formation", "combine_position"]
    sep_nums = ["cushion", "pass_length", "down", "yards_to_go", "yds_to_ez", "in_motion_at_ball_snap", "quarter"]
    exp, info = fit_oe(sep, "separation_at_pass_forward", sep_cats, sep_nums)
    oe_info.append({"model": "separation (WR/TE, all routes)", **info})
    sep = sep.with_columns(pl.Series("sep_oe", sep["separation_at_pass_forward"].to_numpy() - exp))
    # targeted-only separation (smaller n; context of the actual target)
    sep = sep.with_columns(pl.when(pl.col("target")).then(pl.col("separation_at_pass_forward")).alias("sep_tgt"),
                           pl.when(pl.col("target")).then(pl.col("sep_oe")).alias("sep_oe_tgt"))
    # targets per route as a usage outcome
    wr = wr.with_columns(pl.col("target").cast(pl.Int8).alias("tgt_per_route"))
    for g in ("WR", "TE"):
        s = sep.filter(pl.col("combine_position") == g)
        metrics += [(g, s, "separation_at_pass_forward", "separation"), (g, s, "sep_oe", "separation OE"),
                    (g, s, "sep_tgt", "separation (targeted)"), (g, s, "sep_oe_tgt", "separation OE (targeted)"),
                    (g, wr.filter(pl.col("combine_position") == g), "tgt_per_route", "targets/route"),
                    (g, s, "cushion", "cushion faced")]

    # ---------------- DL: get-off, pressure
    dl = df.filter(pl.col("combine_position") == "DL")
    go = dl.filter(pl.col("player_get_off").is_not_null() & (pl.col("player_get_off") < 3)).with_columns(
        pl.col("pass_result").is_not_null().cast(pl.Int8).alias("is_pass"), pl.col("blitzing").cast(pl.Int8))
    go_cats = ["lined_up_position", "offense_formation", "receiver_alignment", "team_coverage_type"]
    go_nums = ["is_pass", "blitzing", "down", "yards_to_go", "yds_to_ez", "quarter", "in_motion_at_ball_snap"]
    exp, info = fit_oe(go, "player_get_off", go_cats, go_nums)
    oe_info.append({"model": "get-off (DL, pass+run)", **info})
    go = go.with_columns(pl.Series("getoff_oe", go["player_get_off"].to_numpy() - exp))
    rush = dl.filter((pl.col("blitzing") == True) & pl.col("pass_result").is_not_null()).with_columns(  # noqa: E712
        pl.col("time_to_pressure").is_not_null().cast(pl.Int8).alias("pressure"),
        (pl.col("sack").fill_null(0) > 0).cast(pl.Int8).alias("sack_any"),
        (pl.col("time_to_pressure").is_not_null() & (pl.col("quick_pressure") == True)).cast(pl.Int8)  # noqa: E712
        .alias("quick_pressure_rate"))
    pr_cats = ["lined_up_position", "offense_formation", "receiver_alignment", "team_coverage_type"]
    pr_nums = ["down", "yards_to_go", "yds_to_ez", "quarter", "in_motion_at_ball_snap"]
    exp, info = fit_oe(rush, "pressure", pr_cats, pr_nums, binary=True)
    oe_info.append({"model": "pressure (DL rush snaps)", **info})
    rush = rush.with_columns(pl.Series("pressure_oe", rush["pressure"].to_numpy() - exp))
    rush_go = rush.filter(pl.col("player_get_off").is_not_null() & (pl.col("player_get_off") < 3))
    metrics += [("DL", go, "player_get_off", "get-off (s)"), ("DL", go, "getoff_oe", "get-off OE"),
                ("DL", rush_go, "player_get_off", "get-off, rush snaps only"),
                ("DL", rush, "pressure", "pressure rate"), ("DL", rush, "pressure_oe", "pressure OE"),
                ("DL", rush, "quick_pressure_rate", "quick-pressure rate"), ("DL", rush, "sack_any", "sack rate"),
                ("DL", rush.filter(pl.col("time_to_pressure").is_not_null()), "time_to_pressure",
                 "time to pressure (pressures only)")]

    # ---------------- OL: pressure probability / pressure allowed
    ol = df.filter((pl.col("combine_position") == "OL") & pl.col("pressure_allowed").is_not_null()).with_columns(
        pl.col("pressure_allowed").cast(pl.Int8), (pl.col("sack_allowed") > 0).cast(pl.Int8).alias("sack_allowed_any"))
    ol_cats = ["lined_up_position", "offense_formation", "receiver_alignment", "team_coverage_type"]
    ol_nums = ["dropback_duration", "pass_rushers_encountered", "down", "yards_to_go", "yds_to_ez", "quarter",
               "in_motion_at_ball_snap"]
    olp = ol.filter(pl.col("peak_pressure_probability_allowed").is_not_null())
    exp, info = fit_oe(olp, "peak_pressure_probability_allowed", ol_cats, ol_nums)
    oe_info.append({"model": "peak pressure prob. allowed (OL)", **info})
    olp = olp.with_columns(pl.Series("ppa_oe", olp["peak_pressure_probability_allowed"].to_numpy() - exp))
    exp, info = fit_oe(ol, "pressure_allowed", ol_cats, ol_nums, binary=True)
    oe_info.append({"model": "pressure allowed (OL)", **info})
    ol = ol.with_columns(pl.Series("pa_oe", ol["pressure_allowed"].to_numpy() - exp))
    metrics += [("OL", olp, "peak_pressure_probability_allowed", "peak pressure prob."),
                ("OL", olp, "ppa_oe", "peak pressure prob. OE"),
                ("OL", ol, "pressure_allowed", "pressure allowed"), ("OL", ol, "pa_oe", "pressure allowed OE"),
                ("OL", ol, "sack_allowed_any", "sack allowed")]

    # ---------------- DB: coverage
    db = df.filter(pl.col("combine_position") == "DB")
    db = db_named_flags(db)
    # validation: does a credited tackle show up as a parenthetical name?
    val = db.filter((pl.col("tackle") == 1) | (pl.col("assist") == 1))
    emit(f"\nDB name-matching check: credited tackle/assist plays with the DB named in (...): "
         f"{val['named_paren'].mean():.3f} (n={val.height})")
    cov = db.filter(pl.col("coverage_assignment").is_not_null() & (pl.col("coverage_assignment") != "NO_ASSIGNMENT")
                    & pl.col("pass_result").is_in(["C", "I", "IN", "S", "R"]))
    thrown = pl.col("pass_result").is_in(["C", "I", "IN"])
    cov = cov.with_columns(
        (thrown & (pl.col("named_paren") | pl.col("named_int"))).cast(pl.Int8).alias("named"),
        (pl.col("coverage_assignment") == "MAN").alias("is_man"),
    ).with_columns(
        (pl.col("named") * pl.col("pass_result").is_in(["I", "IN"]).cast(pl.Int8)).alias("ball_play"),
        pl.col("named_int").cast(pl.Int8).alias("int_rate"),
        pl.when(pl.col("named") == 1).then((pl.col("pass_result") == "C").cast(pl.Int8)).alias("cmp_allowed_named"),
        pl.when(pl.col("named") == 1).then(pl.col("expected_points_added")).alias("epa_named"),
        pl.when(pl.col("is_man") & thrown).then((pl.col("pass_result") == "C").cast(pl.Int8)).alias("team_cmp_man"),
        pl.col("expected_points_added").alias("team_epa_cov"),
    )
    ca = (cov.group_by("coverage_assignment").agg(pl.len().alias("snaps"), pl.col("named").mean().alias("named_rate"),
                                                  pl.col("ball_play").mean().alias("ball_play_rate"),
                                                  pl.col("cmp_allowed_named").mean().alias("cmp_when_named"),
                                                  pl.col("cushion").is_not_null().mean().alias("cushion_rec"))
          .sort("snaps", descending=True))
    emit("\n### DB coverage_assignment values (REG dropbacks, cohort DBs)\n")
    emit(md_table(ca))
    named_exp, info = fit_oe(cov.filter(thrown), "named", ["coverage_assignment", "lined_up_position",
                                                           "team_coverage_type", "receiver_alignment",
                                                           "offense_formation"],
                             ["down", "yards_to_go", "yds_to_ez", "pass_length", "cushion"], binary=True)
    oe_info.append({"model": "DB credited on target (thrown passes)", **info})
    ct = cov.filter(thrown).with_columns(pl.Series("named_oe", cov.filter(thrown)["named"].to_numpy() - named_exp))
    man = cov.filter(pl.col("is_man"))
    metrics += [("DB", cov, "named", "credited-defender rate / cov snap"),
                ("DB", ct, "named_oe", "credited rate OE / thrown"),
                ("DB", cov, "ball_play", "PBU+INT rate / cov snap"),
                ("DB", cov, "cmp_allowed_named", "cmp% allowed when credited"),
                ("DB", cov, "epa_named", "EPA when credited"),
                ("DB", man, "named", "credited rate, MAN snaps"),
                ("DB", man, "team_cmp_man", "team cmp% on MAN snaps"),
                ("DB", cov, "team_epa_cov", "team EPA / cov snap"),
                ("DB", cov, "cushion", "cushion")]

    # ---------------- tracking: in-game speed (Speed Tax), all groups
    spd = tracking_speed()
    sp = df.join(spd, on=["game_id", "play_id", "nfl_id"], how="inner").with_columns(
        pl.col("pass_result").is_not_null().cast(pl.Int8).alias("is_pass"))
    emit(f"\nTracking speed joined to player_play (REG): {sp.height:,} player-plays")
    exp, info = fit_oe(sp, "max_speed", ["combine_position", "lined_up_position", "route_ran", "coverage_assignment",
                                         "offense_formation"],
                       ["is_pass", "down", "yards_to_go", "yds_to_ez", "pass_length"])
    oe_info.append({"model": "max speed per play (all groups)", **info})
    sp = sp.with_columns(pl.Series("max_speed_oe", sp["max_speed"].to_numpy() - exp))
    for g in ("WR", "TE", "DB", "DL", "OL"):
        s = sp.filter(pl.col("combine_position") == g)
        metrics += [(g, s, "max_speed", "max speed/play (trk)"), (g, s, "max_speed_oe", "max speed OE (trk)")]

    # ---------------- sample sizes
    ss_rows, ss_season = [], []
    seen = set()
    for g, d, c, lab in metrics:
        if (g, lab) in seen:
            continue
        seen.add((g, lab))
        r, sr = sample_sizes(d, g, c, lab)
        ss_rows.append(r)
        ss_season += sr
    emit("\n### Per-player sample sizes (REG, all seasons pooled)\n")
    emit(md_table(pl.DataFrame(ss_rows)))
    emit("\n### Per-player-season sample sizes\n")
    sdf = pl.DataFrame(ss_season)
    emit(md_table(sdf.filter(pl.col("outcome").is_in(["separation", "get-off (s)", "pressure rate", "pressure allowed",
                                                       "peak pressure prob.", "credited-defender rate / cov snap",
                                                       "max speed/play (trk)"]))))

    # ---------------- OE models
    emit("\n### Over-expected models (5-fold GroupKFold by player, out-of-fold)\n")
    emit(md_table(pl.DataFrame(oe_info, infer_schema_length=None)))

    # ---------------- reliability
    rel_rows = []
    for g, d, c, lab in metrics:
        rel_rows.append(reliability_row(d, g, c, lab))
    # consistency: per-player SD as the trait
    for g, d, c, lab in [("WR", sep.filter(pl.col("combine_position") == "WR"), "separation_at_pass_forward", "SD separation"),
                         ("WR", sep.filter(pl.col("combine_position") == "WR"), "sep_oe", "SD separation OE"),
                         ("DL", go, "getoff_oe", "SD get-off OE"), ("DL", go, "player_get_off", "SD get-off"),
                         ("OL", olp, "ppa_oe", "SD peak pressure prob. OE"),
                         ("DB", sp.filter(pl.col("combine_position") == "DB"), "max_speed_oe", "SD max speed OE")]:
        rel_rows.append(reliability_row(d, g, c, lab, stat="sd"))
    rel = pl.DataFrame(rel_rows, infer_schema_length=None)
    emit("\n### Reliability: ICC (signal share per play), n for reliability 0.7, split-half Spearman-Brown at min n, "
         "year-over-year r (>=50 per season)\n")
    emit(md_table(rel))

    # split-half curve for headline metrics
    curve = []
    for g, d, c, lab in metrics:
        if lab not in ("separation", "separation OE", "get-off (s)", "get-off OE", "pressure rate", "pressure OE",
                       "peak pressure prob.", "peak pressure prob. OE", "pressure allowed",
                       "credited-defender rate / cov snap", "PBU+INT rate / cov snap", "max speed OE (trk)"):
            continue
        x = d.filter(pl.col(c).is_not_null())
        for m in MIN_NS:
            p, med, r, sb = split_half(x, c, m)
            curve.append({"group": g, "metric": lab, "min_n": m, "players": p, "median_n": med, "r_half": r, "SB": sb})
    emit("\n### Split-half reliability vs minimum sample\n")
    emit(md_table(pl.DataFrame(curve, infer_schema_length=None)))

    # ---------------- per-player outputs with EB shrinkage
    out, out_season = [], []
    for g, d, c, lab in metrics:
        x = d.filter(pl.col(c).is_not_null())
        icc, s2b, s2w = anova_icc(x["nfl_id"].to_numpy(), x[c].to_numpy().astype(float))
        k = s2w / s2b if s2b and s2b > 0 else 1e9
        grand = float(x[c].mean())
        per = x.group_by("nfl_id").agg(pl.len().alias("n"), pl.col(c).mean().alias("mean"))
        per = per.with_columns((pl.col("n") / (pl.col("n") + k)).alias("eb_weight")).with_columns(
            (grand + pl.col("eb_weight") * (pl.col("mean") - grand)).alias("eb_mean"),
            pl.lit(g).alias("group"), pl.lit(lab).alias("metric"))
        out.append(per)
        ps = x.group_by("nfl_id", "season").agg(pl.len().alias("n"), pl.col(c).mean().alias("mean")).with_columns(
            pl.lit(g).alias("group"), pl.lit(lab).alias("metric"))
        out_season.append(ps)
    pl.concat(out, how="vertical_relaxed").write_parquet(PROC / "outcomes_player.parquet")
    pl.concat(out_season, how="vertical_relaxed").write_parquet(PROC / "outcomes_player_season.parquet")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(_md) + "\n")


if __name__ == "__main__":
    sys.exit(main())
