# Confounding by draft slot and selection

Code: `python scripts/eda/confounding.py` (about 2 min, peak RSS about 0.5 GB). All tables below are copied from its output.

**Definitions.**
- Group = `combine_position`. REG only.
- Snaps = `career_offensive_snaps + career_defensive_snaps`. This equals the REG row count in `player_play` for OL/DL/DB (ratio 1.000). **`player_play` holds only pass plays for WR/TE (58% / 40% of career snaps)**, so it cannot be used for WR/TE snap volume. It is fine for NGS per-play fields.
- `snaps_pss` = snaps per available season (3 / 2 / 1 for the 2023 / 2024 / 2025 class).
- `starts_pss` = career games started per available season.
- UDFA pick is imputed as 260, plus a UDFA flag.
- A "player with enough plays" means at least 100 plays with the outcome field present.
- The data has no team snap counts (cohort players only), so "snap share" is `snaps_pss` rescaled. A division by about 1,050 team snaps would not change any correlation.

## 1. Draft slot is the dominant driver of who plays

R^2 is of log(1 + snaps_pss) on log(pick) + UDFA flag + class year.

| grp | n | UDFA share | snaps/season | Spearman (drafted only) | Spearman (UDFA = 260) | R^2 slot | any REG snap | >=100 snaps |
|---|---|---|---|---|---|---|---|---|
| WR | 107 | .28 | 263 | -0.56 | -0.71 | 0.55 | .80 | .65 |
| TE | 42 | .26 | 247 | -0.74 | -0.80 | 0.49 | .81 | .57 |
| DB | 122 | .21 | 278 | -0.68 | -0.73 | 0.43 | .79 | .60 |
| OL | 121 | .26 | 363 | -0.78 | -0.86 | 0.64 | .75 | .57 |
| DL | 118 | .25 | 223 | -0.64 | -0.70 | 0.37 | .83 | .65 |

Share with >=100 REG snaps by round bucket (players per cell in the last table):

| grp | R1 | R2 | R3 | R4-5 | R6-7 | UDFA |
|---|---|---|---|---|---|---|
| WR | 1.00 | 1.00 | .93 | .74 | .65 | .27 |
| TE | - | 1.00 | 1.00 | 1.00 | .00 | .18 |
| DB | 1.00 | 1.00 | .81 | .74 | .36 | .24 |
| OL | 1.00 | 1.00 | .75 | .68 | .38 | .09 |
| DL | 1.00 | 1.00 | .77 | .81 | .41 | .34 |

Mean `snaps_pss` by bucket:

| grp | R1 | R2 | R3 | R4-5 | R6-7 | UDFA |
|---|---|---|---|---|---|---|
| WR | 727 | 469 | 378 | 254 | 226 | 64 |
| TE | - | 461 | 567 | 394 | 24 | 49 |
| DB | 742 | 644 | 515 | 286 | 94 | 40 |
| OL | 905 | 799 | 485 | 256 | 129 | 25 |
| DL | 549 | 482 | 327 | 213 | 97 | 55 |

Players per bucket (R1 / R2 / R3 / R4-5 / R6-7 / UDFA): WR 7/11/14/19/26/30, TE 0/4/4/14/9/11, DB 5/13/16/35/28/25, OL 17/17/12/22/21/32, DL 9/14/13/31/22/29.

Games started per available season falls from about 9-14 (R1) to about 0.4 (UDFA) for every group. Only 9-34% of UDFAs reach 100 snaps, and 22-59% of UDFAs get any snap at all (OL 22%, DB 40%, WR 37%, TE 45%, DL 59%). A model that needs a per-snap outcome sees R1-R3 almost completely and UDFAs hardly ever.

**Rookie ramp and exposure.** There is no strong ramp within a class. Mean snaps by career season (k = 1 / 2 / 3):
- 2023 OL: 401 / 377 / 342
- 2023 DL: 201 / 267 / 270
- 2023 DB: 231 / 262 / 245

Share with >=100 snaps by career season is flat at 0.46-0.53 in every class and season. So dividing by seasons available is reasonable, and the class-year exposure effect is second order. The real exposure problem is **attrition**: cut players stop accruing snaps, so snaps/season of the 2023 class includes zeros that a rookie-only view would not.

## 2. Combine numbers explain little of draft slot, NGS scores explain a lot

Cross-validated R^2 (ridge, 5-fold x 10) of log(pick), all players with UDFA = 260. Drafted-only values are lower for every set:

| grp | size only | stopwatch only | stopwatch + size | NGS scores only | everything |
|---|---|---|---|---|---|
| WR | -0.11 | 0.01 | -0.01 | 0.43 | 0.37 |
| TE | -0.25 | -0.25 | -0.25 | 0.16 | -0.13 |
| DB | -0.12 | 0.11 | 0.09 | 0.50 | 0.55 |
| OL | -0.06 | -0.08 | 0.00 | 0.43 | 0.32 |
| DL | -0.08 | -0.08 | -0.08 | 0.57 | 0.46 |

- Drafted-only CV R^2 for NGS scores is 0.23 (WR), 0.13 (TE), 0.37 (DB), 0.31 (OL), 0.55 (DL).
- Drafted-only CV R^2 for stopwatch + size is at most -0.02, i.e. nothing.
- In-sample adjusted R^2 for stopwatch + size on all players is only 0.16 (WR), 0.24 (DB), 0.18 (OL), ~0 (DL), negative (TE). NGS is 0.44-0.66.
- Predicting **UDFA vs drafted** (CV AUC): stopwatch 0.72 (WR), 0.79 (DB), 0.67 (OL), 0.50 (DL), 0.46 (TE). NGS scores give 0.81 / 0.92 / 0.81 / 0.82 / 0.70.
- Strongest single Spearman with log(pick): `ngs_final_score` at -0.72 to -0.76 in every group, then `ngs_college_production_score` (-0.39 to -0.67), then `forty` (+0.18 to +0.48) and `broad_jump`/`vertical`.
- **`ngs_final_score` and `ngs_college_production_score` are essentially a re-expression of draft capital and college production.** They are contaminated as "athletic" predictors. `ngs_athleticism_score` is the cleaner one (Spearman -0.36 to -0.47).
- Opt-outs: forty/10-yd/vertical/broad are missing for 9-21%. Three-cone, shuttle and bench are missing for 47-80%. Opt-outs are not random with respect to slot (UDFAs are tested less, and high picks skip drills).

## 3. Per-snap outcomes after controlling for slot

Sample: at least 100 plays with the outcome field, controls log(pick), UDFA flag and class year.

| grp | outcome | n (UDFA) | R^2 from slot | R^2 slot + forty + weight + NGS athleticism (complete cases) |
|---|---|---|---|---|
| WR | mean separation at pass forward | 63 (8) | 0.04 | 0.30 (slot-only on same rows 0.04) |
| TE | mean separation | 21 (2) | 0.21 | 0.42 (n=18, slot-only 0.18) |
| OL | mean peak pressure prob. allowed | 66 (3) | 0.015 | 0.07 (slot-only 0.02) |
| OL | pressure_allowed rate | 66 (3) | 0.035 | 0.08 (slot-only 0.05) |
| DL | mean `player_get_off` | 69 (7) | 0.15 | 0.58 (slot-only 0.14) |
| DL | pressure rate | 69 (7) | 0.24 | 0.36 (slot-only 0.21) |
| all | log snaps per season | 42-122 | 0.37-0.64 | +0.01 to +0.14 |

Slot barely moves WR/OL per-snap outcomes (R^2 below 0.05), so those outcomes are close to slot-independent. DL outcomes carry more slot signal. Most of the DL gain comes from physical traits.

Partial correlations, 125 tests, BH-FDR across all of them. 31 have p<.05 (about 6 expected); 18 have q<.05.

| grp | outcome | predictor | partial r | n | q |
|---|---|---|---|---|---|
| DL | get-off | weight | +0.72 | 69 | <.001 |
| DL | get-off | forty | +0.68 | 54 | <.001 |
| DL | get-off | broad jump | -0.69 | 61 | <.001 |
| DL | get-off | vertical | -0.64 | 63 | <.001 |
| DL | pressure rate | ngs_final_score | +0.45 | 69 | .003 |
| DL | pressure rate | forty | -0.41 | 54 | .027 |
| WR | separation | height | -0.53 | 63 | <.001 |
| WR | separation | weight | -0.50 | 63 | .001 |
| OL | pressure prob. allowed | arm length | +0.43 | 66 | .007 |
| WR | snaps | vertical | +0.33 | 96 | .015 |
| TE | snaps | forty | +0.44 | 34 | .088 |

The pattern shows that after controlling for slot, the only strong signals are **mechanical size/role effects**, not skill:
- DL get-off is longer for heavy interior linemen. Adding an interior flag (DT/NT) cuts the partial r from about 0.7 to about 0.5. It survives (q about .003), but n is 54-69 and the DE/DT mix is still imperfectly controlled.
- Small WRs get more separation (r about -0.5 with height and weight).
- OL arm length vs pressure probability drops from 0.43 to 0.14 once G/C vs T is controlled.
- With the role flag, 18 tests have p<.05 and 10 have q<.05 (vs 31 and 18 without).
- **Stopwatch times and NGS athleticism otherwise do not predict per-snap outcomes** beyond slot: WR forty r=-0.29 (q=.16), OL forty r=-0.05, NGS athleticism |r| at most 0.28 (DL pressure rate, q=.12; DL snaps r=0.25, q=.04).
- DB: no NGS per-snap outcome, only snap volume. Nothing there is significant (smallest q=.12).

## 4. Exploratory tracking-feature screen

**This is a screen, not a result.**

Features: per-drill mean over reps of peakSpeed, peakAccel, peakDecel, meanSpeedRetained and cut1_codTime. Drills: up to 6 skill drills with at least 50% coverage per group plus the 40 (WR: dagger, gauntlet, speed out, slant, curl, slot strike; OL: pass-pro mirror, 5-yard wave, pass rush drop, pulls, screen; DL: pass rush, front start wave, run and club, four-bag, body control, hoop; etc.). Controls: log(pick), UDFA flag, stopwatch forty, weight, class year. Players without a stopwatch forty are dropped (this biases toward non-opt-outs).

**267 tests; 12 with p<.05 (13 expected under the null); BH-FDR q<.10: 0.** The smallest q is 0.72.

| grp | outcome | tests | median n | min p | min q | max abs r |
|---|---|---|---|---|---|---|
| WR | separation | 25 | 57 | .034 | .95 | .30 |
| WR | snaps | 25 | 94 | .031 | .95 | .23 |
| TE | snaps | 22 | 34 | .007 | .72 | .54 |
| DB | snaps | 29 | 102 | .142 | .95 | .15 |
| OL | pressure prob. allowed | 25 | 53 | .031 | .95 | .32 |
| OL | pressure_allowed rate | 25 | 53 | .043 | .95 | .30 |
| OL | snaps | 29 | 94 | .009 | .72 | .28 |
| DL | get-off | 29 | 50 | .015 | .80 | .37 |
| DL | pressure rate | 29 | 50 | .009 | .72 | .39 |
| DL | snaps | 29 | 88 | .032 | .95 | .24 |

The raw hits (TE gauntlet `meanSpeedRetained` r=-0.54 with n=30; OL 40 and pass-pro `peakDecel` with snaps; DL four-bag `peakAccel` with pressure rate) are what 267 null tests produce. With n of 50-100 the screen can only detect |r| above about 0.3 (n=50) / 0.2 (n=100). **No incremental value of tracking features beyond slot + forty + weight can be claimed or ruled out at this sample size.**

## 5. Selection-bias handling

**Who is in the per-snap sample.** The sample is about 55-60% of each group: WR 63 / 107, TE 21 / 42, DB 73 / 122, OL 66 / 121, DL 69 / 118. Slot alone predicts membership with CV AUC 0.80-0.88, and adding Combine numbers does not help (0.78-0.89).

Share of the sample that is UDFA, vs UDFA share of the group:

| grp | UDFA in outcome sample | UDFA share of group |
|---|---|---|
| WR | 8 | 28% |
| TE | 2 | 26% |
| DB | 6 (>=100 snaps) | 21% |
| OL | 3 | 26% |
| DL | 7 | 25% |

Positivity is nearly violated for OL (3 UDFAs) and TE (2).

- **IPW.** Logistic propensity for "has an outcome sample" on slot + Combine, clipped at 0.05. Max weight is 4.5-14 and the effective sample size is 14 (TE), 26 (OL), 46-60 (WR/DL/DB). The Combine coefficients barely move for WR and DL: WR separation on forty -1.45 vs -1.27 weighted, DL get-off on forty +0.26 vs +0.24. OL changes a lot (forty -0.013 vs -0.072, noisy, n=53). IPW is usable for WR/DL/DB as a sensitivity check, and not for TE or OL. It also assumes selection on observables (slot is observed), so the real danger, selection on unobserved scouting information, is untouched.
- **Heckman.** Not recommended. There is no credible exclusion restriction (a variable that affects whether a player plays but not how he performs per snap): team need / depth chart is not in the data, and slot is on the causal path to both. With 20-70 observations the inverse Mills ratio is collinear with log(pick).
- **Model availability as an outcome.** This is the most defensible use of the full sample. Use all 510 players, with `snaps_pss`, `starts_pss` and P(>=100 snaps) as outcomes. Slot gives R^2 of 0.37-0.64 here, so **a Combine feature has to add to that**. Test it in a count model: negative binomial / Poisson on REG snaps with `log(seasons available)` as the offset (better: games-in-season), plus log(pick), UDFA flag, group and class year. Tracking features on top of slot should be judged by the change in deviance, not by raw correlations.
- **UDFAs.** Keep them, but do not assign pick 260 and trust it. Always add the UDFA flag. UDFAs are 25% of the cohort but have the least outcome data (any snap: 22-59%), so model them with the availability/hurdle model. A hurdle model with P(any snap) then P(snaps | any) fits this data. For per-snap outcomes they are too few to estimate separately; do not extrapolate a slot effect through them.
- **Class-year exposure.** Use an offset of `log(seasons available)` (3 / 2 / 1) or `log(team games)`. The snap ramp across seasons is flat, so a simple offset is adequate. Add class-year dummies as well: 2025 WR snaps/season is lower (114 vs 192 for 2023 rookies) and OL 2024 is lower (319 vs 401), which shows class composition differs. Do not pool the 2025 class's single season into per-snap outcome means without a minimum-plays rule.

## Implications for project ideas

1. **Stopwatch results and size explain almost none of draft slot** (CV R^2 about 0 in every group; the best is DB at 0.09-0.11). So the Combine's measured athleticism is *not* collinear with draft slot, and tracking features are not simply proxies for it. This makes incremental-value work feasible. The NGS scores are the exception: they are a pre-draft grade that already contains slot information (rho about -0.75) and should not be used as "athleticism" features.
2. **Draft slot explains 37-64% of playing time.** Any model of "Combine predicts NFL success" that does not condition on slot, or does not model availability, is mostly reading draft capital. Use snaps / starts / P(>=100) as outcomes only with log(pick) + UDFA flag as controls.
3. **Which groups have enough non-selected signal.**
   - **DL** is best: 69 in the per-snap sample (7 UDFA), two NGS outcomes whose slot R^2 is only 0.15-0.24, and 118 players for availability. Control for role (DT vs DE); the large get-off effects are mostly size.
   - **WR** is next: 63 (8 UDFA), separation is nearly slot-independent (R^2 0.04). Height/weight role effects dominate and must be controlled.
   - **OL** has an outcome sample that is practically all drafted players (3 UDFAs); NGS pressure outcomes have R^2 below 0.04 from slot, so the sample is low-selection but low-signal. Use the full sample (121) for availability/starts, where slot R^2 is 0.64.
   - **DB**: no direct per-snap NGS outcome. Only availability (122 players) can be modeled, and slot already explains 43%; per-snap coverage outcomes need to be constructed from `coverage_assignment` x target results.
   - **TE** (42 players, 21 with an outcome sample, 2 UDFAs, no first-round TEs) is too small for anything but descriptive work. Pool TE with WR as "receivers" if it must be included.
4. **Tracking-feature screening cannot be confirmed at this n.** 267 tests produced 12 nominal hits, equal to chance, with min q=0.72. Pre-register a small set of features with a physical rationale (for example OL pass-pro mirror lateral speed vs pressure allowed; DL get-off drills vs `player_get_off`) rather than a wide screen. Treat any result as exploratory unless it replicates across classes (2023 vs 2024+2025).
5. **Practical recipe.** For each group: (a) availability model on all players with the offset and slot controls, (b) per-snap outcome model on the sample with slot, UDFA flag, role/position and class controls, (c) report an IPW sensitivity run for WR/DL/DB, (d) report partial R^2 / deviance gain of tracking features over `log(pick) + forty + weight`, with shrinkage (ridge or a few PCA components per drill family) instead of one test per feature.
