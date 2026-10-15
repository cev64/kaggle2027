"""Confounding by draft slot and selection into an outcome sample.

Run:  python scripts/eda/confounding.py   (prints markdown tables; well under 1 GB RAM)

Sections
  1  playing time vs draft slot (REG only, exposure-adjusted)
  2  how much of draft slot do Combine numbers explain (R^2)
  3  per-snap NGS outcomes: baseline R^2 from draft slot, partial corr of stopwatch/NGS/size
  4  exploratory screen of Combine tracking features (BH-FDR)
  5  selection diagnostics: IPW for "has an outcome sample", rookie-year ramp
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
GROUPS = ["WR", "TE", "DB", "OL", "DL"]
LAST_SEASON = 2025
UDFA_PICK = 260  # imputed slot for undrafted players (after pick 257)
SIZE = ["combine_height", "combine_weight", "arm_length", "hand_size", "wing_span"]
STOP = ["forty", "ten_yd_split", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps"]
NGS = ["ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]
MIN_PLAYS = 100  # plays with the outcome field present to enter the per-snap sample


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    import re

    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    is_count = [bool(re.fullmatch(r"n|n_\w+|\w+_n|\w+ n|tests|p_lt_05|outcome_sample|n_udfa", str(c))) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v, cnt in zip(row, is_count):
            if pd.isna(v) if not isinstance(v, str) else False:
                cells.append("")
            elif cnt and not isinstance(v, str):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- data
def load_players() -> pd.DataFrame:
    p = pl.read_csv(RAW / "players.csv", null_values=["NA"])
    c = pl.read_csv(RAW / "combine_results.csv", null_values=["NA"])
    s = pl.read_csv(RAW / "player_career_successes.csv", null_values=["NA"])
    df = p.join(c.drop("draft_year"), on="nfl_id").join(s, on="nfl_id").to_pandas()
    df = df.rename(columns={"combine_position": "grp"})
    df["udfa"] = df.draft_overall_pick.isna().astype(int)
    df["pick_f"] = df.draft_overall_pick.fillna(UDFA_PICK)
    df["lpick"] = np.log(df.pick_f)
    df["round_lab"] = df.draft_round.map(lambda r: "UDFA" if pd.isna(r) else str(int(r)))
    df["n_seasons"] = LAST_SEASON - df.draft_year + 1
    # within-group role flag: interior DL, interior OL (G/C), safety
    df["sub"] = df.nfl_position.isin(["DT", "NT", "G", "C", "FS", "SS", "S"]).astype(int)
    return df


def load_pp() -> pd.DataFrame:
    cols = ["game_id", "play_id", "nfl_id", "separation_at_pass_forward", "pressure_allowed",
            "peak_pressure_probability_allowed", "player_get_off", "time_to_pressure"]
    pp = pl.read_csv(RAW / "player_play.csv", columns=cols, infer_schema_length=10000, null_values=["NA"])
    g = pl.read_csv(RAW / "games.csv").select("game_id", "season", "season_type")
    return pp.join(g, on="game_id").to_pandas()


def player_outcomes(pp: pd.DataFrame, pl_df: pd.DataFrame):
    reg = pp[pp.season_type == "REG"]
    agg = reg.groupby("nfl_id").agg(
        reg_snaps=("play_id", "size"),
        reg_games=("game_id", "nunique"),
        n_sep=("separation_at_pass_forward", "count"),
        sep=("separation_at_pass_forward", "mean"),
        n_off=("player_get_off", "count"),
        getoff=("player_get_off", "mean"),
        n_pb=("peak_pressure_probability_allowed", "count"),
        ppp=("peak_pressure_probability_allowed", "mean"),
        pres_allowed_rate=("pressure_allowed", "mean"),
    )
    rush = reg[reg.player_get_off.notna()]
    agg["dl_pres_rate"] = rush.assign(h=rush.time_to_pressure.notna()).groupby("nfl_id").h.mean()
    df = pl_df.merge(agg, left_on="nfl_id", right_index=True, how="left")
    for c in ["reg_snaps", "reg_games", "n_sep", "n_off", "n_pb"]:
        df[c] = df[c].fillna(0)
    # player_play holds every REG snap for OL/DL/DB (matches career file exactly) but only PASS plays for WR/TE,
    # so total snap volume comes from the career file (offense + defense; it equals REG for OL/DL/DB).
    df["reg_snaps_pp"] = df.reg_snaps
    df["reg_snaps"] = df.career_offensive_snaps + df.career_defensive_snaps
    df["snaps_pss"] = df.reg_snaps / df.n_seasons  # snaps per available season
    df["starts_pss"] = df.career_games_started / df.n_seasons
    df["l_snaps_pss"] = np.log1p(df.snaps_pss)
    df["ge100"] = (df.reg_snaps >= 100).astype(int)
    df["any_snap"] = (df.reg_snaps > 0).astype(int)
    per = reg.groupby(["nfl_id", "season"]).size().rename("n").reset_index()
    return df, per


# ----------------------------------------------------------------------------- stats helpers
def ols_fit(y, X):
    y = np.asarray(y, dtype=float)
    X1 = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    return beta, y - X1 @ beta


def partial_corr(y, x, Z):
    """Partial correlation of x and y given Z (complete cases): r, p, n."""
    d = pd.concat([y.rename("y"), x.rename("x"), Z], axis=1).dropna()
    n = len(d)
    zc = [c for c in d.columns if c not in ("y", "x")]
    if n < len(zc) + 10 or d.x.std() == 0:
        return np.nan, np.nan, n
    Zm = d[zc].values.astype(float)
    r = np.corrcoef(ols_fit(d.y.values, Zm)[1], ols_fit(d.x.values, Zm)[1])[0, 1]
    dof = n - len(zc) - 2
    t = r * np.sqrt(dof / (1 - r**2))
    return r, 2 * stats.t.sf(abs(t), dof), n


def draft_controls(d: pd.DataFrame, cls: bool = True) -> pd.DataFrame:
    Z = pd.DataFrame({"lpick": d.lpick, "udfa": d.udfa}, index=d.index)
    if cls:
        Z["c24"] = (d.draft_year == 2024).astype(int)
        Z["c25"] = (d.draft_year == 2025).astype(int)
    return Z


def r2(y, X):
    res = ols_fit(y, X)[1]
    y = np.asarray(y, dtype=float)
    return 1 - (res**2).sum() / ((y - y.mean()) ** 2).sum()


def cv_r2(y, X, reps=10, seed=0):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 3, 20)))
    return cross_val_score(m, X, y, cv=RepeatedKFold(n_splits=5, n_repeats=reps, random_state=seed), scoring="r2").mean()


def prep_features(d: pd.DataFrame, cols, max_missing=0.6):
    """Mean-impute plus missing indicators; drop columns missing > max_missing within the group."""
    X = pd.DataFrame(index=d.index)
    for c in cols:
        miss = d[c].isna().mean()
        if miss > max_missing or d[c].notna().sum() < 20:
            continue
        X[c] = d[c].fillna(d[c].mean())
        if miss > 0.05:
            X[c + "_na"] = d[c].isna().astype(int)
    return X


# ----------------------------------------------------------------------------- 1
def section1(df):
    out = ["## 1. Draft slot vs playing time (REG)\n"]
    rows = []
    for g in GROUPS:
        d = df[df.grp == g]
        dr = d[d.udfa == 0]
        rows.append(dict(
            grp=g, n=len(d), udfa_share=d.udfa.mean(), mean_snaps_pss=d.snaps_pss.mean(),
            spearman_drafted=stats.spearmanr(dr.draft_overall_pick, dr.snaps_pss).statistic,
            spearman_all=stats.spearmanr(d.pick_f, d.snaps_pss).statistic,
            r2_logsnaps_on_slot=r2(d.l_snaps_pss.values, draft_controls(d).values),
            any_snap=d.any_snap.mean(), ge100=d.ge100.mean()))
    out.append(md(pd.DataFrame(rows).set_index("grp")))
    for g in GROUPS:
        d = df[df.grp == g]
        t = d.groupby("round_lab").agg(n=("nfl_id", "size"), any_snap=("any_snap", "mean"), ge100=("ge100", "mean"),
                                       snaps_pss=("snaps_pss", "mean"), starts_pss=("starts_pss", "mean"))
        t = t.reindex([r for r in ["1", "2", "3", "4", "5", "6", "7", "UDFA"] if r in t.index])
        out.append(f"\n**{g}** by round (snaps_pss = REG snaps per available season, starts_pss = career games "
                   f"started / available seasons)\n")
        out.append(md(t))
    d = df.copy()
    d["bucket"] = pd.cut(d.draft_round.fillna(8), [0, 1, 2, 3, 5, 7, 8], labels=["1", "2", "3", "4-5", "6-7", "UDFA"])
    out.append("\nShare with >=100 REG snaps, by coarse round bucket\n")
    out.append(md(d.pivot_table(index="grp", columns="bucket", values="ge100", aggfunc="mean", observed=False).reindex(GROUPS)))
    out.append("\nPlayers per bucket\n")
    out.append(md(d.pivot_table(index="grp", columns="bucket", values="nfl_id", aggfunc="size", observed=False).reindex(GROUPS), ".0f"))
    out.append("\nMean snaps per available season by coarse bucket\n")
    out.append(md(d.pivot_table(index="grp", columns="bucket", values="snaps_pss", aggfunc="mean", observed=False).reindex(GROUPS), ".0f"))
    return "\n".join(out)


# ----------------------------------------------------------------------------- 2
def section2(df):
    out = ["## 2. How much of draft slot do Combine numbers explain?\n",
           "Targets: log(pick) among drafted, log(pick) with UDFA = 260 (all players), UDFA vs drafted (AUC). "
           "CV R^2 = ridge, 5-fold x10; adj R^2 = OLS on the same mean-imputed design.\n"]
    sets = {"size": SIZE, "stopwatch": STOP, "ngs": NGS, "stop+size": STOP + SIZE, "all": STOP + SIZE + NGS}
    rows = []
    for g in GROUPS:
        d = df[df.grp == g]
        for name, cols in sets.items():
            for tgt, sub in [("drafted_lpick", d[d.udfa == 0]), ("all_lpick", d)]:
                X = prep_features(sub, cols)
                if X.shape[1] == 0:
                    continue
                y = sub.lpick.values
                n, k = X.shape
                r = r2(y, X.values)
                rows.append(dict(grp=g, features=name, target=tgt, n=n, k=k, adj_r2=1 - (1 - r) * (n - 1) / max(n - k - 1, 1),
                                 cv_r2=cv_r2(y, X.values)))
            X = prep_features(d, cols)
            if X.shape[1] == 0:
                continue
            auc = np.mean([roc_auc_score(d.udfa, cross_val_predict(
                make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500)), X.values, d.udfa,
                cv=KFold(5, shuffle=True, random_state=s), method="decision_function")) for s in range(5)])
            rows.append(dict(grp=g, features=name, target="udfa_auc", n=len(d), k=X.shape[1], cv_r2=auc))
    res = pd.DataFrame(rows)
    for tgt in ["drafted_lpick", "all_lpick"]:
        for m, lab in [("cv_r2", "CV R^2"), ("adj_r2", "in-sample adj R^2")]:
            t = res[res.target == tgt].pivot(index="grp", columns="features", values=m).reindex(GROUPS)[list(sets)]
            out.append(f"\n{lab} for {tgt}\n")
            out.append(md(t))
    t = res[res.target == "udfa_auc"].pivot(index="grp", columns="features", values="cv_r2").reindex(GROUPS)[list(sets)]
    out.append("\nCV AUC for UDFA vs drafted (0.5 = chance)\n")
    out.append(md(t))
    rows = []
    for g in GROUPS:
        d = df[df.grp == g]
        for c in STOP + SIZE + NGS:
            m = d[[c, "lpick"]].dropna()
            if len(m) >= 25:
                rows.append(dict(grp=g, var=c, rho=stats.spearmanr(m[c], m.lpick).statistic))
    out.append("\nSpearman of each Combine variable with log(pick), all players (UDFA = 260), pairwise complete\n")
    out.append(md(pd.DataFrame(rows).pivot(index="var", columns="grp", values="rho")[GROUPS]))
    out.append("\nMissing share (opt-outs) by group\n")
    out.append(md(df.groupby("grp")[STOP + NGS].agg(lambda s: s.isna().mean()).T[GROUPS], ".2f"))
    return "\n".join(out)


# ----------------------------------------------------------------------------- 3
OUTCOMES = {
    "WR": [("sep", "n_sep", "mean separation_at_pass_forward")],
    "TE": [("sep", "n_sep", "mean separation_at_pass_forward")],
    "DL": [("getoff", "n_off", "mean player_get_off"), ("dl_pres_rate", "n_off", "pressure rate")],
    "OL": [("ppp", "n_pb", "mean peak_pressure_prob_allowed"), ("pres_allowed_rate", "n_pb", "pressure_allowed rate")],
    "DB": [],
}
PREDS = ["forty", "ten_yd_split", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps",
         "ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score",
         "combine_weight", "combine_height", "arm_length"]


def section3(df):
    out = ["## 3. Per-snap outcomes: what is left after draft slot?\n",
           f"Sample: REG, >= {MIN_PLAYS} plays with the outcome field present. Controls Z = log(pick), UDFA flag, class "
           "year dummies (UDFA pick imputed 260). Partial r on available cases for each predictor.\n"]
    base_rows, pc_rows = [], []
    items = []
    for g in GROUPS:
        items += [(g, col, ncol, label) for col, ncol, label in OUTCOMES[g]]
        items.append((g, "l_snaps_pss", None, "log(1+snaps per avail. season), all players"))
    for g, col, ncol, label in items:
        d = df[df.grp == g]
        if ncol:
            d = d[d[ncol] >= MIN_PLAYS]
        y = d[col]
        full = d.dropna(subset=[col])
        cc = full.dropna(subset=["forty", "combine_weight", "ngs_athleticism_score"])
        r_b = r2(cc[col].values, draft_controls(cc).values) if len(cc) > 15 else np.nan
        r_f = (r2(cc[col].values, np.column_stack([draft_controls(cc).values, cc.forty, cc.combine_weight,
                                                   cc.ngs_athleticism_score])) if len(cc) > 15 else np.nan)
        base_rows.append(dict(grp=g, outcome=label, n=len(full), n_udfa=int(full.udfa.sum()),
                              r2_slot=r2(full[col].values, draft_controls(full).values),
                              n_cc=len(cc), r2_cc_slot=r_b, r2_cc_slot_plus_forty_wt_ngsA=r_f))
        Z = draft_controls(d)
        for pcol in PREDS:
            m = d[[col, pcol]].dropna().astype(float)
            if len(m) < 25:
                continue
            r1, p1, n1 = partial_corr(y, d[pcol], Z)
            rp, pp_, _ = partial_corr(y, d[pcol], Z.assign(sub=d["sub"])) if g in ("DL", "OL", "DB") else (r1, p1, n1)
            pc_rows.append(dict(grp=g, outcome=col if ncol else "snaps", pred=pcol, n=n1,
                                raw_spearman=stats.spearmanr(m[col], m[pcol]).statistic, partial_r=r1, p=p1,
                                partial_r_role=rp, p_role=pp_))
    base = pd.DataFrame(base_rows).set_index("grp")
    out.append("Baseline R^2 of each outcome on draft slot (+ class), and on the complete-case subset with forty, weight, "
               "NGS athleticism added\n")
    out.append(md(base))
    pc = pd.DataFrame(pc_rows).dropna(subset=["p"])
    pc["q_fdr"] = stats.false_discovery_control(pc.p.values, method="bh")
    pc["q_role"] = stats.false_discovery_control(pc.p_role.fillna(1).values, method="bh")
    out.append(f"\nPartial correlations: {len(pc)} tests, {(pc.p < .05).sum()} with p<.05 (chance ~{len(pc) * .05:.0f}), "
               f"{(pc.q_fdr < .05).sum()} with BH-FDR q<.05, {(pc.q_fdr < .10).sum()} with q<.10.\n"
               f"With a sub-position (role) control added for DL/OL/DB (interior DL, G/C, safety): "
               f"{(pc.p_role < .05).sum()} with p<.05, {(pc.q_role < .05).sum()} with q<.05.\n")
    for g in GROUPS:
        sub = pc[pc.grp == g]
        comb = pd.DataFrame(index=[p for p in PREDS if p in set(sub.pred)])
        for o in sub.outcome.unique():
            s = sub[sub.outcome == o].set_index("pred")
            comb[f"{o} r"], comb[f"{o} q"], comb[f"{o} n"] = s.partial_r, s.q_fdr, s.n
            if g in ("DL", "OL", "DB"):
                comb[f"{o} r_role"], comb[f"{o} q_role"] = s.partial_r_role, s.q_role
        comb.index.name = "pred"
        out.append(f"\n**{g}** partial r (q = BH-FDR across all {len(pc)} tests in this section)\n")
        out.append(md(comb))
    out.append("\nSmallest p-values in section 3\n")
    out.append(md(pc.sort_values("p").head(10).set_index("grp")))
    return "\n".join(out)


# ----------------------------------------------------------------------------- 4
CANON = {"PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR", "PASS_PRO_MIRROR_DRILL": "PASS_PRO_MIRROR",
         "LINE": "LINE_DRILL", "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",
         "SPEED_OUT_ROUTE_LEFT": "SPEED_OUT_LEFT"}
TFEATS = ["peakSpeed", "peakAccel", "peakDecel", "meanSpeedRetained", "cut1_codTime"]


def section4(df):
    out = ["## 4. Exploratory incremental-value screen (tracking features)\n"]
    t = pl.read_parquet(PROC / "combine_drill_summary.parquet").to_pandas()
    t["drill"] = t.drillName.str.upper().replace(CANON)
    t = t.merge(df[["nfl_id", "grp"]], left_on="nflId", right_on="nfl_id")
    t = t[(t.drillType == "SKILL_DRILLS_" + t.grp) | (t.drillType == "FORTY_YARD_DASH")]
    t["drill"] = np.where(t.drillType.str.startswith("SKILL"), t.grp + ":" + t.drill, t.drillType)
    results, cover = [], []
    for g in GROUPS:
        gp = t[t.grp == g]
        n_g = (df.grp == g).sum()
        cov = gp.groupby("drill").nfl_id.nunique().sort_values(ascending=False)
        main = [d for d in cov.index if d.startswith(g + ":") and cov[d] >= 0.5 * n_g][:6] + ["FORTY_YARD_DASH"]
        cover.append(dict(grp=g, main_drills_with_n_players=", ".join(f"{m.split(':')[-1]} ({cov[m]})" for m in main)))
        wide = gp[gp.drill.isin(main)].groupby(["nfl_id", "drill"])[TFEATS].mean().unstack("drill")
        wide.columns = [f"{d.split(':')[-1]}|{f}" for f, d in wide.columns]
        d = df[df.grp == g].set_index("nfl_id").join(wide)
        outs = [(c, n) for c, n, _ in OUTCOMES[g]] + [("l_snaps_pss", None)]
        for col, ncol in outs:
            dd = d[d[ncol] >= MIN_PLAYS] if ncol else d
            dd = dd[dd.forty.notna()]
            Z = pd.DataFrame({"lpick": dd.lpick, "udfa": dd.udfa, "forty": dd.forty, "wt": dd.combine_weight,
                              "c24": (dd.draft_year == 2024).astype(int), "c25": (dd.draft_year == 2025).astype(int)})
            for f in wide.columns:
                if dd[f].notna().sum() < 30:
                    continue
                r, p, n = partial_corr(dd[col], dd[f], Z)
                results.append(dict(grp=g, outcome=col if ncol else "snaps", feature=f, n=n, partial_r=r, p=p))
    res = pd.DataFrame(results).dropna(subset=["p"])
    res["q_fdr"] = stats.false_discovery_control(res.p.values, method="bh")
    out.append(md(pd.DataFrame(cover).set_index("grp")))
    out.append(f"\nControls: log(pick), UDFA flag, stopwatch forty, weight, class year; players with no stopwatch forty "
               f"dropped. Features: mean over reps of {', '.join(TFEATS)} per drill (cut1_codTime only where defined).\n")
    out.append(f"**{len(res)} tests; p<.05: {(res.p < .05).sum()} (chance expectation {len(res) * .05:.0f}); "
               f"BH-FDR q<.10: {(res.q_fdr < .10).sum()}; q<.05: {(res.q_fdr < .05).sum()}.**\n")
    s = res.groupby(["grp", "outcome"]).agg(tests=("p", "size"), n_median=("n", "median"), p_lt_05=("p", lambda x: (x < .05).sum()),
                                           min_p=("p", "min"), min_q=("q_fdr", "min"), max_abs_r=("partial_r", lambda x: x.abs().max()))
    out.append(md(s))
    out.append("\nTop 12 by p-value\n")
    out.append(md(res.sort_values("p").head(12).set_index("grp")))
    return "\n".join(out)


# ----------------------------------------------------------------------------- 5
def section5(df, per):
    out = ["## 5. Selection diagnostics\n"]
    df = df.copy()
    ncol = {"WR": "n_sep", "TE": "n_sep", "DL": "n_off", "OL": "n_pb"}
    df["has_out"] = df.ge100.astype(float)  # DB: no NGS per-snap outcome -> >=100 snaps
    for g, c in ncol.items():
        df.loc[df.grp == g, "has_out"] = (df.loc[df.grp == g, c] >= MIN_PLAYS).astype(float)
    rows, ipw_rows = [], []
    for g in GROUPS:
        d = df[df.grp == g]
        Xs = prep_features(d, ["forty", "combine_weight", "combine_height", "arm_length", "ngs_athleticism_score",
                               "ngs_college_production_score"])
        base = draft_controls(d)
        y = d.has_out.values.astype(int)
        auc = {}
        for name, X in [("slot", base), ("slot+combine", pd.concat([base, Xs], axis=1))]:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            auc[name] = np.mean([roc_auc_score(y, cross_val_predict(m, X.values, y, cv=KFold(5, shuffle=True, random_state=s),
                                                                     method="decision_function")) for s in range(5)])
        Xf = pd.concat([base, Xs], axis=1)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(Xf.values, y)
        ph = pd.Series(np.clip(m.predict_proba(Xf.values)[:, 1], 0.05, 1), index=d.index)
        w = 1 / ph[y == 1]
        rows.append(dict(grp=g, n=len(d), outcome_sample=int(y.sum()), share=y.mean(),
                         udfa_share=d.udfa.mean(), udfa_with_sample=d.has_out[d.udfa == 1].mean(),
                         drafted_with_sample=d.has_out[d.udfa == 0].mean(),
                         auc_slot=auc["slot"], auc_slot_combine=auc["slot+combine"],
                         ipw_max=w.max(), ipw_ess=w.sum() ** 2 / (w**2).sum()))
        for col, nc, _ in OUTCOMES[g][:2]:
            sub = d[d[nc] >= MIN_PLAYS]
            for pred in ["forty", "ngs_athleticism_score"]:
                s2 = sub.dropna(subset=[col, pred])
                if len(s2) < 25:
                    continue
                Z = np.column_stack([s2.lpick, s2.udfa, s2.draft_year == 2024, s2.draft_year == 2025, s2[pred]]).astype(float)
                sw = np.sqrt(1 / ph.loc[s2.index].values)
                X1 = np.column_stack([np.ones(len(s2)), Z]) * sw[:, None]
                ipw_rows.append(dict(grp=g, outcome=col, pred=pred, n=len(s2),
                                     beta_unweighted=ols_fit(s2[col].values, Z)[0][-1],
                                     beta_ipw=np.linalg.lstsq(X1, s2[col].values.astype(float) * sw, rcond=None)[0][-1],
                                     sd_y=s2[col].std(), sd_x=s2[pred].std()))
    out.append("Who gets an outcome sample (WR/TE >=100 routes with separation, DL >=100 rush plays, OL >=100 pass-block "
               "plays, DB >=100 REG snaps), and CV AUC of slot / slot + Combine for predicting it\n")
    out.append(md(pd.DataFrame(rows).set_index("grp")))
    out.append("\nIPW (logistic propensity on slot + Combine, clipped at 0.05) vs unweighted coefficient on the Combine "
               "variable, controlling slot and class\n")
    out.append(md(pd.DataFrame(ipw_rows).set_index("grp"), ".4f"))
    c = df[["nfl_id", "draft_year", "grp"]].merge(per, on="nfl_id", how="left")
    c["k"] = c.season - c.draft_year + 1
    full = df[["nfl_id", "draft_year", "grp"]].merge(pd.DataFrame({"k": [1, 2, 3]}), how="cross")
    full = full[full.draft_year + full.k - 1 <= LAST_SEASON]
    full = full.merge(c[["nfl_id", "k", "n"]].dropna(), on=["nfl_id", "k"], how="left").fillna({"n": 0})
    out.append("\nMean REG snaps by career season k (zeros included), by group and class: rookie ramp / exposure\n")
    out.append(md(full.pivot_table(index=["grp", "draft_year"], columns="k", values="n", aggfunc="mean"), ".0f"))
    out.append("\nShare with >=100 REG snaps in career season k, by class (all groups)\n")
    out.append(md(full.assign(ge=(full.n >= 100) * 1).pivot_table(index="draft_year", columns="k", values="ge", aggfunc="mean")))
    d = df.assign(career=df.career_offensive_snaps + df.career_defensive_snaps)
    qa = pd.DataFrame({g: {"corr_log": np.corrcoef(np.log1p(x.career), np.log1p(x.reg_snaps_pp))[0, 1],
                           "pp_over_career": x.reg_snaps_pp.sum() / x.career.sum()} for g, x in d.groupby("grp")}).T
    out.append("\nQA: REG rows in player_play vs career offensive+defensive snaps (WR/TE rows in player_play are pass plays only)\n")
    out.append(md(qa))
    return "\n".join(out)


def main():
    df = load_players()
    df, per = player_outcomes(load_pp(), df)
    print(f"players: {len(df)}; REG snaps total {int(df.reg_snaps.sum())}\n")
    for f in (lambda: section1(df), lambda: section2(df), lambda: section3(df), lambda: section4(df), lambda: section5(df, per)):
        print(f(), "\n", flush=True)


if __name__ == "__main__":
    main()
