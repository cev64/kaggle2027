# Drill coverage and kinematic quality, by position group

Produced by `scripts/eda/drill_quality.py` (`PYTHONPATH=src python scripts/eda/drill_quality.py [--refresh]`, about 2 min, peak RSS 0.6 GB). It writes the per-attempt features to `data/processed/eda/attempt_features.parquet` (6,310 attempts) and every table below as a CSV in `data/processed/eda/`. Figures are in `docs/eda/fig/dq_*.png`. Groups come from `combine_results.combine_position`. Throughout, ICC means ICC(3,1) on each player's first two reps (by `attempt`), after removing flagged outlier reps.

**Summary**
- Coverage is broad but shallow. Every group has 6–13 skill drills covering at least 90% of its players in all three years. Only the 40 and one or two drills per group have a second rep: GAUNTLET (WR, TE), the DB 45° drill and LINE_DRILL (DB), PASS_RUSH_DROP (OL) and PASS_RUSH_DRILL (DL). WR route breaks, the OL mirror and wave drills, and the DB 90° and W drills are single-rep.
- Raw kinematics are clean: `s` matches the speed derived from positions with r = 0.9994. The problems are trimming and frames. Clips start already moving and end while the player is still at speed, and the clip length changes by year for several drills (2024 DB 45° clips are half as long). The `normalize_drill` heading is set from the first 2 yd, so lateral-start drills get random axes.
- The 40 is excellent once the start is anchored on distance. t(2→40 yd) gives r = 0.98 against the official 40, and the tracked 10–40 split matches the official forty − ten_yd_split with r = 0.98 and a bias of −0.05 s. Times measured from the start of the clip, including the 10-yd split, are poor (r = 0.47 for the 10-yd split).
- Reliable features that the stopwatch does not explain: the shape of the 40 curve (speed at 10 yd relative to peak; distance to 90% of top speed) and the OL mirror reversal features. Route-break and DB-transition mechanics are not reliable.

## 1. Coverage by group and drill

"Usable" means at least 80 players, or at least 85% of the group, and present in all three years. Drills with fewer than 10 players are omitted: the four DL players who ran SKILL_DRILLS_LB, and the DL WAVE/FOUR_BAG_SHUFFLE/SHUFFLE_SPRINT drills. "2+ reps" counts players with at least two attempts at that drill.

**WR (n=107)**

| drill | players | attempts | 2+ reps | years | usable |
|---|---|---|---|---|---|
| SPEED_OUT_ROUTE_LEFT | 107 | 124 | 17 | 23/24/25 | yes |
| SLANT_ROUTE_LEFT | 107 | 110 | 3 | 23/24/25 | yes |
| GAUNTLET_DRILL | 107 | 200 | 93 | 23/24/25 | yes |
| DAGGER_ROUTE_LEFT | 107 | 115 | 8 | 23/24/25 | yes |
| CURL_ROUTE_RIGHT | 103 | 110 | 7 | 23/24/25 | yes |
| SLOT_STRIKE_ROUTE_LEFT | 101 | 102 | 1 | 23/24/25 | yes |
| OVER_SHOULDER_ADJUST | 100 | 101 | 1 | 23/24/25 | yes |
| GO_ROUTE_RIGHT | 98 | 105 | 7 | 23/24/25 | yes |
| SLOT_SAIL_ROUTE_RIGHT | 98 | 106 | 8 | 23/24/25 | yes |
| RED_ZONE_FADE_RIGHT | 98 | 112 | 13 | 23/24/25 | yes |
| COMEBACK_ROUTE_RIGHT | 98 | 110 | 12 | 23/24/25 | yes |
| POST_CORNER_ROUTE_RIGHT | 96 | 111 | 15 | 23/24/25 | yes |
| FORTY_YARD_DASH | 94 | 176 | 82 | 23/24/25 | yes |
| POST_CORNER_ROUTE_LEFT | 70 | 80 | 10 | 23/24 |  |
| CIRCUS_ROUTE_LEFT | 37 | 43 | 6 | 25 |  |
| SHORT_SHUTTLE | 29 | 36 | 6 | 23/24/25 |  |
| THREE_CONE_DRILL | 24 | 30 | 5 | 23/24/25 |  |

**TE (n=42)**

| drill | players | attempts | 2+ reps | years | usable |
|---|---|---|---|---|---|
| FLAT_ROUTE_LEFT | 42 | 42 | 0 | 23/24/25 | yes |
| OVER_SHOULDER_ADJUST | 42 | 46 | 4 | 23/24/25 | yes |
| HOOK_ROUTE_RIGHT | 41 | 45 | 4 | 23/24/25 | yes |
| GAUNTLET_DRILL | 41 | 81 | 40 | 23/24/25 | yes |
| IN_ROUTE_LEFT | 40 | 43 | 3 | 23/24/25 | yes |
| WHEEL_ROUTE_LEFT | 40 | 45 | 5 | 23/24/25 | yes |
| BLOCK_EXPLOSION | 39 | 40 | 1 | 23/24/25 | yes |
| FORTY_YARD_DASH | 34 | 66 | 32 | 23/24/25 | borderline (81%) |
| CORNER_ROUTE | 30 | 61 | 29 | 23/24/25 |  |
| END_ZONE_FADE_RIGHT | 28 | 35 | 7 | 24/25 |  |
| SHORT_SHUTTLE | 21 | 37 | 11 | 23/24/25 |  |
| THREE_CONE_DRILL | 19 | 27 | 7 | 23/24/25 |  |

**DB (n=122)**

| drill | players | attempts | 2+ reps | years | usable |
|---|---|---|---|---|---|
| BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL | 122 | 124 | 2 | 23/24/25 | yes |
| BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION | 122 | 241 | 119 | 23/24/25 | yes |
| BOX_DRILL | 122 | 125 | 3 | 23/24/25 | yes |
| W_DRILL | 121 | 124 | 3 | 23/24/25 | yes |
| GAUNTLET_DRILL | 119 | 123 | 4 | 23/24/25 | yes |
| LINE_DRILL | 118 | 217 | 99 | 23/24/25 | yes |
| TERYL_AUSTIN_DRILL_2 | 111 | 113 | 2 | 23/24/25 | yes |
| FORTY_YARD_DASH | 102 | 195 | 93 | 23/24/25 | yes |
| TERYL_AUSTIN_DRILL_1 | 101 | 103 | 2 | 23/24/25 | yes |
| SHORT_SHUTTLE | 28 | 36 | 8 | 23/24/25 |  |
| THREE_CONE_DRILL | 24 | 31 | 7 | 23/24/25 |  |

**OL (n=121)**

| drill | players | attempts | 2+ reps | years | usable |
|---|---|---|---|---|---|
| PASS_PRO_MIRROR_DRILL | 121 | 121 | 0 | 23/24/25 | yes |
| PASS_RUSH_DROP | 120 | 239 | 118 | 23/24/25 | yes |
| FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE | 120 | 120 | 0 | 23/24/25 | yes |
| SCREEN_DRILL | 119 | 120 | 1 | 23/24/25 | yes |
| OL_PULL_DRILL_FOLD_BLOCK_RIGHT | 119 | 119 | 0 | 23/24/25 | yes |
| OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT | 98 | 98 | 0 | 23/24/25 | yes |
| FORTY_YARD_DASH | 95 | 179 | 84 | 23/24/25 | yes |
| OL_PULL_DRILL_LONG_PULL_RIGHT | 93 | 93 | 0 | 23/24/25 | yes |
| SHORT_SHUTTLE | 54 | 70 | 13 | 23/24/25 |  |
| THREE_CONE_DRILL | 48 | 55 | 7 | 23/24/25 |  |

**DL (n=118)**

| drill | players | attempts | 2+ reps | years | usable |
|---|---|---|---|---|---|
| PASS_RUSH_DRILL | 118 | 225 | 106 | 23/24/25 | yes |
| FOUR_BAG_AGILITY_DRILL | 112 | 112 | 0 | 23/24/25 | yes |
| FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION | 112 | 112 | 0 | 23/24/25 | yes |
| RUN_AND_CLUB_DRILL | 112 | 113 | 1 | 23/24/25 | yes |
| BODY_CONTROL_DRILL | 111 | 113 | 2 | 23/24/25 | yes |
| RUN_THE_HOOP_DRILL | 109 | 119 | 10 | 23/24/25 | yes |
| FORTY_YARD_DASH | 93 | 177 | 84 | 23/24/25 | yes |
| BACK_PEDAL_AND_REACT | 59 | 62 | 3 | 23/24/25 |  |
| THREE_CONE_DRILL | 45 | 60 | 12 | 23/24/25 |  |
| SHORT_ZONE_BREAKS | 40 | 119 | 39 | 23/24 |  |
| SHORT_SHUTTLE | 38 | 44 | 4 | 23/24/25 |  |
| SHORT_ZONE_BREAK_FORWARD | 20 | 20 | 0 | 25 |  |
| SHORT_ZONE_BREAK_FLAT_WHEEL | 20 | 20 | 0 | 25 |  |
| SHORT_ZONE_BREAK_INSIDE | 18 | 18 | 0 | 25 |  |


Notes. The tracked 3-cone and shuttle cover only 20–45% of each group, so they are not usable as group-wide features. The official times cover more players (3-cone 18–61 per group, shuttle 20–64). DL `SHORT_ZONE_BREAKS` (2023–24) and its three 2025 sub-drills cannot be pooled without care. 4,986 player-drill cells have 1 rep in 75% of cases, 2 reps in 24% and 3 or more in 1%.

## 2. Kinematic quality

**Reported `s` vs speed derived from positions** (central difference of x, y over 418k player frames):

| check | value |
|---|---|
| r(s, position speed), lag 0 / ±1 frame | **0.9994** / 0.997 / 0.995 (no time offset) |
| median bias, median \|diff\|, p95, p99.9 | −0.02, 0.05, 0.19, 0.74 yd/s |
| worst drill (3-cone, shuttle) | r 0.993–0.995, p99 \|diff\| 0.6–0.8 yd/s (hard plants) |
| ratio by speed | 0.983 at 0–2 yd/s, 0.999 at >8 yd/s |
| `a` | unsigned total acceleration: r 0.985 with \|a vector\|, 0.14 with signed ds/dt. Use smoothed ds/dt for braking. |
| `dis` | **not** the per-frame step: median dis/step = 0.46, 10–90% range 0.10–0.83. Do not use it; use positions. |

**`dir` noise at low speed** (|dir − heading from positions|; frame-to-frame jitter):

| s (yd/s) | 0–0.25 | 0.25–0.5 | 0.5–1 | 1–2 | 2–4 | 4–6 | >6 |
|---|---|---|---|---|---|---|---|
| median error (deg) | 20.8 | 9.3 | 4.4 | 2.2 | 1.0 | 0.5 | 0.3 |
| p90 error (deg) | 90 | 29 | 14 | 6.5 | 3.1 | 1.5 | 0.9 |
| p90 jitter (deg/frame) | 102 | 83 | 34 | 17 | 9.7 | 5.8 | 3.6 |

Mask `dir` below 1 yd/s. This matters at plants, reversals and mirror dead points, where speed passes through about 0. Take heading from the smoothed velocity and gate on speed.

**Trimming** (`dq_dir_noise_trim.png`):
- **Starts are triggered at motion.** 48% of clips start already moving (s0 > 0.5 yd/s). The mean number of standing frames before onset is 0.2 in the 40 and up to 7–9 in GAUNTLET/SCREEN. Onset-relative times (tTo3, tTo5, t10) therefore inherit an undefined start.
- **Ends are cut at speed.** 97% of clips end above 50% of peak speed, and 41% have their peak in the last 3 frames, so there is no stopping phase. For OL pull, PASS_RUSH_DROP, RUN_THE_HOOP and BOX, the last frame is the peak in at least 86% of clips. `peakSpeed` there is "speed at the cut-off point", and post-drill decel cannot be measured.
- About 4% of clips end with ≥2 frames of exactly constant velocity, i.e. extrapolated padding: 6% in 2023–24 and 0.3% in 2025.
- **Clip length is year-dependent** (median frames or path length differs by more than 20% across years; `year_consistency.csv`):

| drill | metric | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION | frames | 116 | **60** | 101 |
| SLANT_ROUTE_LEFT | path yd | 23.9 | 24.3 | **13.3** |
| FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE | frames | 158 | 161 | **122** |
| FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION | path yd | 42.6 | 51.1 | **30.7** |
| WHEEL_ROUTE_LEFT / COMEBACK_ROUTE_RIGHT | path yd | 41 / 27 | 39 / 26 | **29 / 32** |

Year-centring changes conclusions. The DB 45° drill's duration ICC falls from 0.91 to 0.04 and its peak decel ICC from 0.77 to 0.01. Any drill feature must be centred (or modelled) by draft year.

**Geometry after `normalize_drill`** (`dq_paths.png`; `geometry.csv`). The axis comes from the first 2 yd of movement, which is mostly stance wobble:
- The 40 ends a median of 5.5° off-axis (p90 12°). GO routes are 4.9° off.
- Lateral-start drills get arbitrary axes: OL mirror endpoint angle IQR 58°, 5-yd wave 36°, DL front-start wave 98°.
- DB 90° break, DB 45°, PASS_RUSH_DRILL, BACK_PEDAL drills and TERYL_AUSTIN break left or right across attempts. Use `mirror=True`, or the sign of the main turn.

Where a break exists, its location is tight:

| drill | median distance start→break | IQR | turn angle (median, IQR) |
|---|---|---|---|
| CURL | 11.7 yd | 1.4 | 132°, 22 |
| COMEBACK | 17.9 yd | 1.4 | 116°, 17 |
| W_DRILL | 3.3 yd | 0.5 | 124°, 88 (several breaks) |
| DB 90° break | 19.1 yd | 4.4 | 80°, 13 |
| DB 45° (late break) | 48.1 yd | 5.4 | 135°, 8 |

Recommended frames: for straight-line drills, use the start→end chord (the 40's run length is then 40.4 ± 0.5 yd). For lateral drills, use the principal axis of the shuffle phase.

**Break detection.** `bdb27.drills.detect_cuts` measures the turn between the entry and the exit *speed peak*. On routes the exit peak comes after the receiver turns upfield, so it finds a break in only 26% of curls, 27% of daggers, 14% of in-routes and 2% of slants. The script's `detect_breaks` measures exit heading 0.5 s after the speed minimum and raises curls to 92% and in-routes to 98%. Slants (2%), speed-outs (15%) and post-corners (4%) have no speed minimum (prominence < 0.75 yd/s), so `heading_break` (20%→80% of a target heading change) handles them.

**Outlier attempts** (robust z > 4 within drill; `outlier_flags.parquet`): 209 of 6,310 (3.3%).
- 81 are truncated or aborted (short path or duration): 34 are the 2025 slant clips (a year trim, not aborted reps), 12 are shuttles with fewer than 2 turns and 5 are 3-cones.
- 112 are overlong, mostly route reps with extra jogging after the catch, plus PASS_RUSH_DROP and pull drills.
- 26 are slow reps: 14 three-cone, 3 shuttle.
- 40 tracked 3-cone and 22 shuttle clips are more than 1 s off the official time because a turn is missing or there is pre-drill movement.
- No duplicate reps: the 375 forty rep pairs are distinct runs a median of 18 min apart.

## 3. Validity against the stopwatch

The best tracked rep per player is compared with the official time. Bias = tracking − official. r is computed after dropping clips more than 1 s off (counts shown). `dq_stopwatch.png`.

| official | tracking measure | n | dropped | r | bias (s) | SD diff (s) | within-group r (WR/TE/DB/OL/DL) |
|---|---|---|---|---|---|---|---|
| forty | t(40 yd) from first frame | 418 | 0 | 0.905 | +0.30 | 0.135 | .41/.53/.47/.72/.90 |
| forty | clip duration | 418 | 0 | 0.878 | +0.33 | 0.155 | .40/.51/.44/.65/.85 |
| forty | **t(2→40 yd)** | 418 | 0 | **0.981** | −0.53 | **0.065** | **.80/.88/.81/.90/.98** |
| forty − ten_yd_split | **t(10→40 yd)** | 418 | 0 | **0.979** | **−0.045** | **0.044** | .86/.87/.86/.96/.99 |
| ten_yd_split | t(10 yd) from first frame | 418 | 0 | 0.467 | +0.33 | 0.133 | .03/.13/.04/.31/.43 |
| ten_yd_split | t(2→10 yd) | 418 | 0 | 0.794 | −0.48 | 0.073 | .01/.24/.22/.54/.74 |
| three_cone | onset (s = 0.5) → clip end | 142 | 4 | 0.896 | +0.43 | 0.173 | .58/.70/.79/.86/.87 |
| short_shuttle | onset → clip end | 154 | 4 | 0.608 | +0.58 | 0.235 | .30/.05/.30/.54/.79 |
| short_shuttle | first turn → clip end | 154 | 1 | 0.832 | −1.28 | 0.135 | .57/.08/.70/.78/.73 |

Reading:
1. From 10 yd onward the tracking and the official 40 agree to within 0.04 s. The whole +0.3 s bias, and almost all the disagreement, comes from the first 2 yd: the clip starts about 0.3 s before the official clock.
2. The official 10-yd split has almost no within-group signal against tracking: r ≈ 0 within WR. The tracked t(2→10) has an ICC of 0.77, so the official split is the noisier of the two.
3. The 3-cone is valid (r 0.90) on the 30% of players who have it. The full shuttle is not (r 0.61), because its start is badly trimmed; first-turn→finish is fine (r 0.83).

## 4. Reliability

### 4a. Test-retest (two reps), year-centred ICC(3,1)

These are the features most relevant to the ideas. The full table, including raw ICCs, is in `reliability.csv`; see also `dq_reliability.png`.

| drill (pairs) | feature: ICC |
|---|---|
| FORTY (374) | peakSpeed **0.99**, t(10→20) 0.98, t(20→40) 0.99, t(10→40) **0.99**, v10 0.92, t40 0.83, t(2→10) **0.77**, d90 (distance to 90% of vmax) **0.71**, v5 0.68, v10/vmax **0.64**, peakAccel 0.45, tTo5 0.36, t10 from clip 0.38 |
| GAUNTLET WR/TE (108–132) | end-break entry speed 0.67, peak decel 0.65, min speed 0.57, speed retained 0.60, peak re-accel 0.61, exit speed 0.47, decel time 0.00, decel distance −0.01; peakSpeed 0.65 |
| DB 45° REACTION (63–119) | peakSpeed 0.50, tTo5 0.50, pedal (entry) speed 0.50, **transition turn time 0.13, peak yaw rate 0.06**, late-break min speed −0.07, COD time −0.05, peak decel −0.03, duration 0.04 (0.91 raw) |
| DL PASS_RUSH_DRILL (100) | peakSpeed 0.75, duration 0.48, peakDecel 0.32, **tTo3 0.10, tTo5 0.20, peakAccel 0.18** |
| DB LINE_DRILL (98) | tTo5 0.56, peakSpeed 0.40, peakAccel 0.12, duration −0.31 |
| OL PASS_RUSH_DROP (101–105) | peakSpeed 0.52, tTo5 0.33, peakAccel 0.21, peakDecel −0.06 |
| 3-cone / shuttle (30 / 35) | peakDecel 0.51 / 0.73, peakAccel 0.61 / 0.65, total time 0.37 / 0.53 |
| WR routes (10–16 pairs only) | speed-out entry speed 0.74 (16 pairs), post-corner peakAccel 0.81 (15); everything else is too small to judge |

Pattern: speeds at a fixed distance and peak speeds are reliable. Onset-timed quantities (tTo3, tTo5, t10) and decel duration or distance are not. Peak accel and decel come in between (0.2–0.7).

### 4b. Single-rep drills: split-half within the attempt (lateral drills)

Odd vs even reversals, Spearman-Brown corrected. "Tempo-partialled" removes the attempt's reversal rate (the leader's tempo, which both halves share) and draft year. `split_half_reversals.csv`; `dq_mirror_lateral.png`.

| drill | feature | SB raw | SB tempo-partialled | r with tempo | consecutive-player autocorr |
|---|---|---|---|---|---|
| OL PASS_PRO_MIRROR | dead time (\|v\| < 1 yd/s at reversal) | 0.89 | **0.84** | +0.43 | 0.08 |
| OL PASS_PRO_MIRROR | peak lateral decel | 0.83 | **0.76** | −0.41 | 0.13 |
| OL PASS_PRO_MIRROR | lateral peak speed | 0.88 | 0.73 | −0.68 | 0.32 |
| OL PASS_PRO_MIRROR | overshoot (peak → zero crossing) | 0.85 | 0.64 | −0.71 | 0.33 |
| OL PASS_PRO_MIRROR | brake time (90% → 0) | 0.71 | 0.38 | −0.68 | 0.31 |
| OL PASS_PRO_MIRROR | re-accel time (0 → 2 yd/s) | 0.54 | 0.50 | +0.17 | 0.26 |
| OL 5-yd WAVE | lateral peak speed / overshoot / dead time / peak decel | 0.59 / 0.31 / 0.35 / −0.48 | 0.51 / 0.02 / 0.36 / −0.29 | | 0.55 / 0.42 / 0.11 / 0.10 |
| DL FRONT-START WAVE | lateral peak speed / overshoot / dead time / peak decel | 0.49 / 0.29 / 0.51 / 0.41 | 0.38 / 0.31 / 0.19 / 0.10 | | 0.60 / 0.72 / −0.05 / 0.01 |

Split-half inside a single attempt is an upper bound, because day, partner and fatigue effects are shared. Cross-drill agreement between the OL mirror and the 5-yd wave is about 0 (r −0.05 to 0.13). The mirror features are internally consistent, but their stability across days is untested.

### 4c. Single-rep drills: parallel forms across drills (year-centred, `cross_drill.csv`)

| set | feature | k drills | median pairwise r | standardized alpha |
|---|---|---|---|---|
| WR route breaks (7–9 routes, n ≈ 95) | break entry speed (heading-based) | 7 | 0.17 | 0.61 |
| | peakAccel | 9 | 0.18 | 0.62 |
| | cut peak decel / mean decel | 5 | 0.07 / 0.10 | 0.34 / 0.34 |
| | cut min speed / speed retained / decel time / COD time | 5 / 5 / 5 / 3 | 0.04 / 0.02 / −0.03 / −0.11 | 0.02 / 0.04 / −0.02 / −0.11 |
| | heading-break turn time / peak re-accel | 7 | 0.07 / −0.02 | 0.29 / −0.13 |
| DB breaks (6 drills, n ≈ 110) | peakSpeed | 6 | 0.14 | 0.55 |
| | cut entry / min speed / peak decel / COD time | 6 / 6 / 6 / 5 | 0.08 / 0.00 / 0.01 / 0.07 | 0.30 / 0.21 / −0.10 / 0.16 |
| DL (6 drills, n ≈ 110) | peakSpeed | 6 | 0.64 | **0.91** |
| | peakAccel / tTo3 / tTo5 | 6 | 0.21 / 0.04 / 0.09 | 0.61 / 0.27 / 0.36 |

Braking and COD quality do not behave like a player trait across routes or DB drills. Only speed-level features (entry speed, peak speed) aggregate.

### 4d. Consistency (rep-to-rep variability as a trait)

For each player, |rep1 − rep2| was turned into a within-drill percentile rank. Ranks were averaged over random halves of the drills and the two halves correlated (409 players with 2+ drills that have 2 reps): duration r = −0.03, peakSpeed −0.02, peakAccel 0.03, COD time 0.01 (21 players). **No player-level consistency signal.** With 1% of cells having 3 or more reps, within-player SD cannot be estimated.

## 5. Redundancy with stopwatch and NGS

Player-mean features within each group. R² comes from regressing the feature on forty + weight. "Unique reliable" = reliability − R², an approximate upper bound on reliable variance the stopwatch does not explain. Full table: `redundancy.csv`.

| group | drill: feature | r forty | r 3-cone (n) | r shuttle (n) | r NGS athl. | R² forty+wt | reliability | unique reliable |
|---|---|---|---|---|---|---|---|---|
| WR | FORTY peakSpeed | −0.83 | −0.26 (30) | 0.02 (35) | 0.53 | 0.68 | 0.99 | 0.30 |
| WR | FORTY t(10→40) | 0.91 | 0.27 | 0.09 | −0.64 | 0.83 | 0.99 | 0.16 |
| WR | FORTY t(2→10) | 0.28 | −0.06 | −0.01 | −0.28 | 0.09 | 0.77 | **0.68** |
| WR | FORTY v10/vmax | 0.10 | 0.06 | −0.18 | 0.02 | 0.01 | 0.64 | **0.63** |
| WR | FORTY d90 | −0.07 | −0.00 | 0.21 | −0.06 | 0.01 | 0.71 | **0.71** |
| DB | FORTY t(2→10) / v10/vmax / d90 | 0.38 / 0.10 / −0.00 | | | −0.46 / 0.07 / −0.10 | 0.18 / 0.01 / 0.00 | .77/.64/.71 | **.59/.63/.71** |
| OL | FORTY t(2→10) / v10/vmax / d90 | 0.56 / 0.30 / −0.31 | 0.42 / −0.04 / −0.02 (59) | | −0.54 / −0.13 / 0.15 | 0.31 / 0.10 / 0.12 | | .46/.54/.60 |
| DL | FORTY t(2→10) / v10/vmax / d90 | 0.79 / 0.61 / −0.62 | 0.55 / 0.44 / −0.40 (38) | | −0.36 / −0.21 / 0.24 | 0.63 / 0.38 / 0.38 | | .14/.27/.33 |
| DL | FORTY peakSpeed | −0.98 | −0.67 | −0.67 | 0.45 | 0.96 | 0.99 | 0.03 |
| DL | PASS_RUSH peakSpeed / tTo5 / peakAccel | −0.75 / 0.34 / −0.31 | −0.63 / 0.32 / −0.14 | | 0.33 / −0.02 / 0.05 | 0.57 / 0.12 / 0.13 | .75/.20/.18 | .18/.08/.05 |
| OL | MIRROR dead time | 0.20 | 0.17 (61) | 0.22 (64) | −0.12 | 0.08 | 0.84* | **0.76** |
| OL | MIRROR peak lat. decel | −0.22 | −0.21 | −0.18 | 0.16 | 0.07 | 0.76* | **0.69** |
| OL | MIRROR overshoot | −0.04 | −0.11 | 0.11 | −0.05 | 0.00 | 0.64* | 0.64 |
| WR | route breaks (curl/comeback/out/slant/dagger) | \|r\| ≤ 0.31 | \|r\| ≤ 0.39 | \|r\| ≤ 0.36 | \|r\| ≤ 0.24 | ≤ 0.10 | low (4c) | — |
| DB | 45° transition turn time / yaw | −0.07 / 0.01 | | −0.11 / −0.40 | 0.01 / 0.06 | ≤ 0.02 | 0.13 / 0.06 | ~0.1 |

\* Within-attempt split-half, tempo-partialled (an upper bound).

Takeaways:
- 40 peak speed and the 10–40 split mostly re-express the forty (R² 0.65–0.96). The early-acceleration shape does not: t(2→10), v10/vmax and d90 are reliable and nearly orthogonal to forty and weight in WR, DB and OL.
- DL speed is almost entirely the forty plus body weight.
- Route and DB break features are "unique" only because they are mostly noise.
- `ngs_athleticism_score` tracks the forty (|r| 0.45–0.74) and does not cover acceleration shape.

## Implications for project ideas

| idea | drill-side support | reasons |
|---|---|---|
| **Speed Tax** (40 speed and acceleration curve) | **strong** | The 40 is tracked for 418/510 (79–88% per group, 84% of TE fall to 81%), with 2 reps for 375. It is validated against the stopwatch: t(2→40) r = 0.98; the 10–40 split matches the official to within 0.04 s. Peak speed ICC is 0.99 and the splits 0.98–0.99. The curve-shape features (t(2→10) 0.77, d90 0.71, v10/vmax 0.64) are reliable and nearly independent of forty and weight in WR/DB/OL (R² ≤ 0.31). Caveats: drop the first 2 yd (the clip start is 0.3 s before the official start); peak speed alone is largely redundant with the forty. |
| **DL burst** | **ok (40-based) / weak (skill drills)** | The 40's acceleration features are reliable, but for DL they are largely forty + weight (R² 0.38–0.63). PASS_RUSH_DRILL has 2 reps for 100, but its onset-based burst (tTo3, tTo5, peakAccel) has ICC 0.10–0.20 because clip starts are triggered inconsistently. Its peak speed (0.75) is forty-redundant (R² 0.57). FRONT_START_WAVE reversal features are unreliable (≤ 0.38). A distance-anchored burst from the 40, t(2→10), is the usable signal. |
| **Mirror Image** (OL) | **ok** | PASS_PRO_MIRROR covers 121/121 OL in all years but has 1 rep. Within-attempt split-half, after removing tempo, is 0.64–0.84 for dead time, peak lateral decel, lateral peak speed and overshoot. These are close to orthogonal to forty, 3-cone, shuttle, weight and NGS (\|r\| ≤ 0.25). Caveats: no leader is tracked, so true reaction latency cannot be measured, only the player's own reversal mechanics. Tempo explains much of the variance (r ≈ ±0.7) and consecutive players share it (autocorr 0.3). Agreement with the 5-yd wave is about 0, so test-retest across days is unproven. Requires the PCA lateral axis and year centring. |
| **Brake Check** (WR breaks, DB 90°, W) | **weak** | Coverage is excellent (96–107 WR per route, 121–122 DB), but routes are single-rep (3–17 pairs). Break mechanics do not generalise across routes (alpha 0.02–0.34 for min speed, speed retained, decel, COD time); only entry speed and accel aggregate (alpha about 0.6). The repo's `detect_cuts` misses most route breaks, and the slant clip is trimmed short in 2025. The one replicated break, the GAUNTLET end turn, is moderately reliable (0.57–0.67) but is not a route break. DB break features have parallel-form r ≈ 0. Best option: a route-composite "entry speed / accel" score, which is closer to speed than to braking skill. |
| **Trigger Time** (DB 45°) | **weak** | It has 2 reps for 119 DBs, but the 2024 clips are half length. The 45° transition has almost no speed loss and its timing comes from the coach's cue. Transition turn time ICC is 0.13, yaw rate 0.06, and the late-break decel and COD features ≤ 0.07 after year-centring. Only pedal speed and peak speed (about 0.5) carry player signal. There is no stimulus to time a reaction from. |
| **Consistency** (rep-to-rep variability) | **weak** | 75% of player-drill cells have a single rep and 1% have 3 or more. Rep-to-rep differences show no player-level consistency across drills (split-half r −0.03 to 0.03). |

**Practical rules for any drill feature:** anchor on distance, not clip time. Centre on drill × year. Use the chord or PCA axis instead of the first-2-yd heading. Mask `dir` below 1 yd/s. Use `detect_breaks` or `heading_break` (scripts/eda) rather than `drills.detect_cuts` for routes. Drop the 3.3% of flagged reps.

Figures: `fig/dq_paths.png` (normalized paths), `fig/dq_dir_noise_trim.png` (dir error by speed, start and end speeds), `fig/dq_stopwatch.png` (tracking vs official), `fig/dq_reliability.png` (ICC bars), `fig/dq_mirror_lateral.png` (mirror lateral velocity and the detected sprint-out).
