# Outcome metrics: sample size, over-expected models, reliability

_Generated 2026-10-15 by `scripts/eda/outcome_reliability.py` (REG season only, cohort players only). The full tables are in `data/processed/cache/outcome_reliability_tables.md`. Per-player metrics with n, raw mean and an EB-shrunk mean are in `data/processed/outcomes_player.parquet`, with a per-season version in `outcomes_player_season.parquet`._

**Methods**
- **Opportunity definitions**
  - WR/TE: a route run with non-null separation.
  - DL get-off: a non-null `player_get_off` on pass or run plays.
  - DL rush snap: `blitzing == True` on a dropback. For DL this flag means "rushed the passer".
  - DL pressure: `time_to_pressure` is non-null on a rush snap.
  - OL: a non-null `pressure_allowed`, or a non-null `peak_pressure_probability_allowed`.
  - DB coverage snap: `coverage_assignment` is not null and not `NO_ASSIGNMENT`, on a dropback (C/I/IN/S/R).
- **OE (over expected)** = observed minus the out-of-fold prediction of a HistGradientBoosting model, using 5-fold GroupKFold by player. "Additive" is the same model with depth 1 (main effects only). No player identity enters the model.
- **ICC** is the one-way random-effects ICC(1) (method of moments, unequal n), i.e. the per-play between-player variance share. `n_rel70` = (0.7/0.3)·(1−ICC)/ICC is the number of plays needed for reliability 0.7. The EB shrink constant is k = σ²_within/σ²_between, and the shrunk mean is grand + n/(n+k)·(mean − grand).
- **Split-half** splits each player's chronological plays into odd and even plays, takes the Pearson r of the two half-means, and applies Spearman-Brown (SB). Players are included if they have at least min_n plays.
- **YoY** is the Pearson r of season means for consecutive seasons, with at least 50 plays in both seasons.

## 1. Sample sizes (per player, REG, pooled 2023–25)

| group | outcome | plays | players | median n | p25–p75 | ≥50 | ≥100 | ≥200 |
|---|---|---|---|---|---|---|---|---|
| WR (83) | separation (all routes) | 31,035 | 83 | 250 | 121–576 | 69 | 63 | 45 |
| WR | separation, targeted only | 6,513 | 79 | 44 | 19–125 | 36 | 24 | 10 |
| TE (33) | separation (all routes) | 6,781 | 33 | 157 | 26–316 | 22 | 21 | 11 |
| TE | separation, targeted only | 1,325 | 28 | 31 | 14–71 | 12 | 5 | 0 |
| DL (98) | get-off (all snaps) | 33,417 | 98 | 195 | 74–460 | 82 | 69 | 49 |
| DL | rush snaps (pressure / sack denominator) | 27,361 | 98 | 145 | 61–351 | 79 | 65 | 42 |
| DL | time to pressure (pressures only) | 2,703 | 90 | 11 | 5–35 | 17 | 7 | 1 |
| OL (89) | pressure allowed / sack allowed | 48,698 | 89 | 433 | 99–787 | 70 | 66 | 60 |
| OL | peak pressure probability allowed | 46,493 | 89 | 401 | 93–749 | 70 | 66 | 58 |
| DB (92) | coverage snaps | 38,659 | 92 | 267 | 84–688 | 73 | 67 | 59 |
| DB | MAN coverage snaps | 10,032 | 89 | 64 | 17–157 | 52 | 35 | 15 |
| DB | "credited" targets (see §5) | 3,976 | 90 | 29 | 9–70 | 31 | 12 | 0 |
| all | tracking max speed / play | 248,856 | 401 | — | — | 335 | 303 | 260 |

Only about 80% of each Combine group ever records a REG outcome (for example, 83 of 107 WRs). Players per season with at least 100 opportunities:

| outcome | 2023 | 2024 | 2025 |
|---|---|---|---|
| WR separation | 22 | 35 | 45 |
| TE separation | 4 | 11 | 13 |
| DL get-off | 13 | 35 | 55 |
| DL rush snaps | 11 | 29 | 51 |
| OL pressure allowed | 17 | 38 | 56 |
| DB coverage snaps | 17 | 37 | 56 |

The 2023 class contributes seasons in 2023–25, and the 2025 class contributes one season. Single-season samples are therefore typical, and YoY pairs number about 50 per group (about 18 for TE).

## 2. Over-expected models (out-of-fold, grouped by player)

| model | features | n | R² additive | R² GBM | AUC |
|---|---|---|---|---|---|
| WR/TE separation | route, cushion, alignment, lined-up pos, coverage man/zone + type, formation, pass_length, down, ytg, field pos, motion, quarter | 37,816 | 0.262 | **0.311** | |
| DL get-off | lined-up pos (EDGE/IL), pass vs run, rush flag, formation, receiver alignment, coverage type, down, ytg, field pos | 33,417 | 0.225 | **0.240** | |
| DL pressure (binary) | same minus get-off inputs | 27,361 | 0.010 | 0.008 | 0.589 |
| OL peak pressure prob. | **dropback_duration**, **pass_rushers_encountered**, lined-up pos, formation, alignment, coverage type, down, ytg, field pos | 46,493 | 0.183 | **0.191** | |
| OL pressure allowed (binary) | same | 48,698 | 0.088 | 0.097 | 0.783 |
| DB credited on thrown pass | coverage_assignment, lined-up pos, coverage type, alignment, pass_length, cushion, down/ytg | 34,091 | 0.010 | 0.030 | 0.652 |
| max speed / play (tracking) | position group, lined-up pos, route, coverage assignment, pass/run, pass_length, down/ytg | 248,856 | 0.624 | 0.652 | |

Interactions add little. GBM beats the additive model by ≤0.05 R², so a GLM with these main effects would be an adequate expected model. Screens (6.4 yd) and flats (4.7 yd) dominate the separation context. Go routes are lowest (2.6 yd).

## 3. Reliability

| group | metric | ICC/play | n for rel .7 | EB k | SB (≥50) | SB (≥100) | SB (≥200) | YoY r (pairs) |
|---|---|---|---|---|---|---|---|---|
| WR | separation raw | .024 | 96 | 41 | .94 | .95 | .92 | **.79** (56) |
| WR | separation OE | .005 | 430 | 184 | .72 | .79 | .73 | .42 (56) |
| WR | separation raw, targeted | .057 | 39 | 17 | .89 | .92 | .88 | .69 (23) |
| WR | separation OE, targeted | .007 | 349 | 150 | .44 | .22 | .18 | .04 (23) |
| WR | targets / route | .017 | 132 | 56 | .67 | .76 | .88 | .59 (57) |
| TE | separation raw | .009 | 257 | 110 | .84 | .84 | .86 | .49 (18) |
| TE | separation OE | .004 | 546 | 234 | .67 | .66 | .89 (11 pl) | .07 (18) |
| DL | get-off, all snaps | .097 | 22 | 9 | .95 | .96 | .98 | **.80** (54) |
| DL | **get-off, rush snaps** | **.143** | **14** | 6 | .98 | .98 | .99 | **.86** (49) |
| DL | get-off OE | .058 | 38 | 16 | .89 | .92 | .95 | .68 (54) |
| DL | pressure rate | .013 | 172 | 74 | .83 | .83 | .83 | .58 (49) |
| DL | pressure OE | .006 | 362 | 155 | .73 | .72 | .71 | .43 (49) |
| DL | quick-pressure rate | .009 | 244 | 105 | .64 | .72 | .87 | .48 |
| DL | sack rate | .001 | 1,696 | 727 | .24 | .17 | .40 | .44 |
| OL | **peak pressure prob. raw** | .019 | 119 | 51 | .88 | .91 | .90 | **.69** (49) |
| OL | peak pressure prob. OE | .008 | 300 | 129 | .78 | .83 | .81 | .36 (49) |
| OL | pressure allowed | .005 | 448 | 192 | .77 | .79 | .75 | .50 (49) |
| OL | pressure allowed OE | .004 | 601 | 257 | .71 | .73 | .63 | .37 |
| OL | sack allowed | .001 | 2,796 | 1,198 | .34 | .55 | .59 | .10 |
| DB | credited-defender rate / cov snap | .004 | 661 | 283 | .65 | .53 | .64 | .36 (52) |
| DB | PBU+INT rate / cov snap | .002 | 942 | 404 | −.07 | .08 | .41 | .33 |
| DB | cmp% allowed when credited | .024 | 95 | 41 | .68 (31 pl) | .85 (12 pl) | — | — |
| DB | team cmp% on own MAN snaps | .001 | 2,716 | 1,164 | .27 | .18 | .21 | −.13 |
| DB | team EPA / coverage snap | .001 | 1,813 | 777 | .26 | .35 | .50 | −.09 |
| DB | cushion (alignment, not outcome) | .138 | 15 | 6 | .98 | .99 | .98 | .57 |
| WR / TE / DB / DL / OL | max speed OE (tracking) | .038 / .048 / .014 / .036 / .036 | 59 / 47 / 161 / 62 / 63 | | .91 / .94 / .84 / .89 / .91 | .94 / .94 / .87 / .92 / .97 | | .77 / .75 / .57 / .68 / .79 |

Reliability of **play-to-play SD** (the "Consistency" trait), as split-half SB at min n of 50 / 100 / 200:

| SD of | SB at 50 / 100 / 200 |
|---|---|
| WR separation | .77 / .82 / .83 |
| WR separation OE | .50 / .50 / .70 |
| DL get-off | .43 / .59 / .75 |
| DL get-off OE | .50 / .63 / .77 |
| OL peak pressure prob. OE | .61 / .74 / .75 |
| DB max speed OE | .68 / .74 / .72 |

Split-half SB at "≥50" is the reliability of the **whole career sample** of the included players (median n is about 200–550). It is not the reliability at n = 50; for that, use `n_rel70`.

Odd and even plays share games, teams and opponents, so split-half overstates reliability whenever context is persistent. The gap between raw and OE shows how much this matters. WR raw separation has YoY .79, but OE has .42 and the per-play ICC falls from .024 to .005. Roughly **80% of the stable "separation" signal is role and usage** (screen/flat-heavy slot players versus outside go-route receivers), not skill. The same pattern holds for OL peak pressure: raw YoY .69 versus OE .36, because a long dropback or a tackle's alignment is persistent context. DL get-off keeps most of its signal after adjustment (YoY .80 → .68). It is the only NGS outcome whose OE version is reliable at modest n.

## 4. Recommended outcome per group

| group | primary outcome | min-sample filter | players surviving | shrinkage (k) | notes |
|---|---|---|---|---|---|
| **DL** | get-off on rush snaps (raw or alignment-adjusted OE) | ≥50 rush snaps | **76** | 6 (raw) / 16 (OE) | rel ≥ .9 at the filter; YoY .86. Secondary: pressure rate, ≥100 rush snaps → 65 players, k = 74 (n_rel70 = 172) |
| **WR** | separation OE, all routes (raw separation as a robustness check) | ≥100 routes | **63** | 184 (OE) / 41 (raw) | OE rel ≈ .79 at the filter (median 369 routes). Targeted-only separation is too sparse (24 WRs ≥100) and its OE is not reliable |
| **OL** | peak pressure probability allowed, adjusted for dropback duration and rushers encountered | ≥200 pass-block snaps | **58** | 129 (OE) / 51 (raw) | binary pressure_allowed needs n ≈ 450; sack_allowed is noise |
| **TE** | separation (pool with WR in one route-level model) | ≥100 routes | 21 | 234 (OE) | too few for a standalone TE analysis; YoY OE ≈ 0 |
| **DB** | no reliable NGS coverage outcome. Best available: credited-defender rate / coverage snap (heavily shrunk), or in-game max-speed OE | ≥200 coverage snaps | 59 | 283 | coverage results reach rel ≈ .6 only with a full season or more of snaps; team results (cmp%/EPA on the DB's snaps) have ICC ≈ .001 |

**EB shrinkage.** Use the shrunk mean grand + n/(n + k)·(mean − grand) with the k values above, which are already computed in `outcomes_player.parquet`. Examples:
- A WR with 300 routes keeps 62% of his OE deviation.
- An OL with 400 snaps keeps 76% of his peak pressure OE deviation.
- A DL with 150 rush snaps keeps 96% of his get-off deviation.

For downstream regressions, either weight players by n/(n + k) or model plays directly with a player random effect (lme4-style). This beats hard filters, which drop about 25% of each group. For pressure, the binomial EB (beta prior fit by method of moments) is the natural choice. A linear k is used here for comparability.

## 5. Surprises and data notes

- **Separation is recorded on non-targeted routes.** 85% of non-targeted routes have it, versus 100% of targeted routes. It is missing **only on sacks (S) and scrambles (R)**, where there is no pass forward: 100% of C/I/IN routes have it and 0% of S/R routes do. Targeted routes average 3.77 yd versus 2.99 yd for non-targeted routes. Nullified plays never appear in player_play (`play_nullified_by_penalty` is always N).
- **`pass_length` is the air yards of the play's actual throw, not the receiver's route depth.** For non-targeted receivers it is only play context. A route-depth proxy has to come from tracking (the receiver's downfield x at pass-forward).
- **DL `blitzing` is the "rushed the passer" flag.** On DL dropback snaps it is True 89% of the time, False 5% (dropped into coverage) and null 6%, and it is null on runs. `player_get_off` exists on run plays as well (8k REG DL plays). It is capped at 2.0 s.
- **`time_to_pressure` and `quick_pressure` exist only on pressure plays.** The DL pressure rate per rush snap is 9.9%. Time-to-pressure as a trait is unusable: the median DL has 11 pressures, and only 17 have ≥50.
- **DL pressure is almost unpredictable from context** (AUC .59) because no OL or dropback info is attached to DL rows. Joining play-level `dropback_duration` from OL rows of the opposing team would only work for plays that have a cohort OL.
- **The DB "targeted defender" can be inferred from `play_description`.** The DB's "F.Last" appears in the parenthetical after a thrown pass (tackler on completions, pass defender on incompletions) or in "INTERCEPTED by". Validation: credited tackles and assists match 98.4% of the time (n = 5,829). DBs are credited on 10.8% of MAN snaps; when credited, they allow 74.6% completions on MAN snaps versus 81–95% on zone snaps. This is a *credited* defender, not the *covering* defender: on completions it is whoever tackled. NGS has no coverage-matchup field.
- **`coverage_assignment` has 21 values:** MAN, NO_ASSIGNMENT (blitzers and run plays), 18 zone landmarks (DEEP_THIRD_L/M/R, QUARTERS_INSIDE/OUTSIDE_L/R, DEEP_HALF_L/R, CURL_FLAT_L/R, FLAT_L/R, HOOK_CURL_L/R, HOLE, DEEP_FREE) and PREVENT. It differs from `coverage_assignment_at_snap` on 16% of player-plays (post-snap rotation).
- **DB `cushion` is only recorded when the DB is across from a receiver.** It covers 97% of DEEP_THIRD_L/R and 87% of MAN snaps, but only 3% of DEEP_THIRD_MIDDLE and 1% of DEEP_FREE snaps. It mostly measures alignment: ICC .14 and the highest per-play reliability of any field, so it is not a performance outcome.
- Team-level results on a DB's snaps (completion %, EPA) have essentially **zero** between-player variance, and their YoY r is negative. With 1–3 cohort DBs per play, 11 defenders and QB play dilute any individual effect.
- In-game **max speed per play** is extremely role-driven (R² .65 from context alone). Its OE version is still very stable (YoY .68–.79, except DB at .57).

## Implications for project ideas

Ratings are for the **outcome side only** (is there a reliable, adequately sampled target?), on a 5-point scale.

| idea | outcome-side support | reasoning |
|---|---|---|
| **First Step** (DL burst → get-off/pressure) | ★★★★★ get-off; ★★★ pressure | Get-off on rush snaps is the most reliable NGS outcome in the data (ICC .14, n_rel70 = 14, YoY .86), with 76 DL ≥50 rush snaps. Pressure rate is usable with shrinkage (n_rel70 = 172, 65 DL ≥100). Sacks are noise. |
| **Speed Tax** (40 speed → in-game speed/usage) | ★★★★☆ | Tracking max speed is available for about 400 players, is highly reliable and is stable YoY. It must be role-adjusted (R² .65 from context). Usage (routes, snaps, targets/route) is fully observed, but survivorship (no REG snaps for about 20% of players) needs a hurdle or selection model. |
| **Brake Check** (WR/DB break → separation) | ★★★☆☆ WR side; ★☆☆☆☆ DB side | WR separation OE: 63 WRs at rel ≈ .79, but only about 20% of raw separation's stable signal survives context adjustment (YoY .42), so effects will be small. Targeted-only separation is too sparse. There is no DB separation-allowed field, and the opponent receiver is not in the data. |
| **Mirror Image** (OL → pressure allowed) | ★★★☆☆ | Peak pressure probability is a continuous, well-sampled outcome (58 OL ≥200 snaps, rel ≈ .8 OE). Binary pressure allowed needs about 450 snaps per player and sacks allowed are unusable. Adjusting for dropback and rushers halves YoY stability (.69 → .36), so use OE with EB shrinkage. |
| **Consistency** (rep variability → play-to-play variance) | ★★☆☆☆ | Per-player SD reaches SB ≈ .7–.8 only at ≥200 plays (45–58 players per group). SDs of OE metrics are noisier still (.50–.63 at ≥100). Feasible for WR separation and OL/DL with large-n players only. Heteroscedastic mixed models (a player-level residual variance) are better than raw SDs. |
| **Trigger Time** (DB reaction → coverage results) | ★☆☆☆☆ | No reliable DB coverage outcome. Credited-defender rate needs about 660 snaps for rel .7, PBU+INT about 940, and team cmp%/EPA has ICC ≈ 0. Usable only with a tracking-built DB outcome, e.g. post-snap reaction or closing speed from the DB's own trajectory. Even then there is no ball or receiver in game tracking, so results would come only from the description-credited plays (median 29 per DB). |
