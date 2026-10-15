# Tracking data QA report

_Generated 2026-10-15 by `scripts/build_processed.py`._

## Key facts about the real data (checked 2026-10-15 against the downloaded files)

**Files.** 9 CSVs, 2.15 GiB. `combine_tracking` has 463,189 rows. `game_tracking_2023/2024/2025` have 3.67M / 7.83M / 11.16M rows. `player_play` has **314,197** rows (the Kaggle page says 316,338; `wc -l` gives more because `play_description` contains embedded newlines). `games` has 1,002 rows. `players`, `combine_results` and `player_career_successes` have 510 rows each.

**Cohort.** 510 prospects: 160 from 2023, 177 from 2024, 173 from 2025. 127 are UDFAs (null draft pick). There are exactly five Combine position groups: DB 122, OL 121, DL 118, WR 107, TE 42. There are no RB, QB or LB groups. `nfl_position` gives the NFL roster position (17 values, e.g. CB/SS/FS, G/T/C, DE/DT/OLB).

**Combine tracking.**
- One `event_id` is one player's attempt at one drill. It contains a single prospect, never a partner.
- 6,310 attempts across all 510 players. Sampling is exactly 10 Hz (all steps are 100 ms). There are no gaps, duplicates or nulls.
- A `BALL` entity appears on WR/TE route and catch drills, and carries the **receiver's `nfl_id`**.
- `attempt` counts reps *within a `drill_type` session* (1–17 for WR), not within a drill.
- Most player × drill cells have 1 rep. Of the 4,986 player-drill cells, 3,723 have 1, 1,213 have 2 and 50 have 3–5.
- Every prospect has the position-specific `SKILL_DRILLS_<GROUP>`, about 6–12 drills each.
- Stopwatch drills are tracked only for a subset: `FORTY_YARD_DASH` for 418 players, `THREE_CONE_DRILL` for 160, `SHORT_SHUTTLE` for 170.
- Players move up to 11.85 yd/s (99.99th percentile 11.6). The extreme `s`/`a`/`dis` values (up to 25.8 yd/s) are all ball rows.
- Drill names drift across years. These are canonicalized in `drillCanon`: `PASS_PRO-MIRROR_DRILL`/`PASS_PRO_MIRROR_DRILL`, `LINE`/`LINE_DRILL`, `OVER_THE_SHOULDER_ADJUST`/`OVER_SHOULDER_ADJUST`, `SPEED_OUT_LEFT`/`SPEED_OUT_ROUTE_LEFT`.
- Some drills exist only in some years: `PASS_PRO_MIRROR` covers all 3 years but is only 121 OL attempts; DL `SHORT_ZONE_BREAKS` (2023–24) was split into three named breaks in 2025.
- Coordinates are not field-anchored (y spans −11 to 66), so use drill-relative frames (`xd`, `yd`).

**Game tracking.**
- Cohort players only. There are no teammates outside the cohort, no opponents and no ball. All `nfl_id`s are in `players.csv`.
- Up to 10 cohort players share a play; 2023 plays have mostly 1–2.
- Each player-play starts **1.0 s (10 frames) before `ball_snap`** and runs a median of 6.9 s.
- 10 Hz with 6 one-frame gaps. Includes orientation `o`.
- About 0.4% of player-plays have no `ball_snap` and 50 have two.
- One game (2025101205) has 82 rows with negative `a` (signed acceleration). These are set to |a| and flagged `aSignFixed`.
- 99.2% of tracked player-plays appear in `player_play`.

**Season types.** `games.csv` and both play-level files **include preseason and postseason**: 49 PRE, 272 REG and 13 POST games per season. PRE is 15–20% of player-plays (2023: 10.4k PRE vs 38.8k REG in player_play). Filter `seasonType == "REG"` to match the prompt. The 2025 season runs through Super Bowl LX (Feb 2026), so the 2025 class has one full season.

**player_play.** 64 columns, 0 duplicate keys. Two columns drift from the brief's names: `team_abbr` is the player's team, and `visitor_team_win_probility_added` is misspelled in the source. NGS outcome fields are position-specific:
- `separation_at_pass_forward`: WR/TE on about 88% of routes run (targeted or not), 38k WR and 8.7k TE player-plays.
- `cushion`: DB 50k, WR 41k.
- `player_get_off`: DL 43k (+ 690 DB blitzes).
- `peak_pressure_probability_allowed`: OL 57.6k pass-block snaps.
- `pressure_allowed` and `sack_allowed`: OL 60k.
- `time_to_pressure` and `quick_pressure`: DL about 3.5k (only plays where the player generated pressure).
- `coverage_assignment`: DB.
- `route_ran`: WR/TE (12 route types).

DB outcomes have no direct NGS "coverage success" field. Coverage results must be built from `coverage_assignment` × pass result/target, or from the player's own tracking.

**QA interpretation.**
- *Orientation* flags (o vs dir > 90° while running) come from OL kick-slides and DB backpedals: 2023 has 1,382 OL and 1,308 DB player-plays of 3,598. The games flagged for orientation have a median offset of about 0°. This is legitimate movement, not the historic NGS orientation bug, so no correction is applied.
- *Out of bounds* rows are players beyond the sideline or end line after the whistle.
- *Teleports* affect about 0.3% of plays.
- The LOS derived from `yardline_*` checks out against alignment. At the snap, tackles sit a median of 1.3 yd behind the LOS, TEs 1.7 yd and slot WRs 1.6 yd; CBs sit 5.1 yd in front and high safeties 12.5 yd.


## Dataset profile

### Files

| file | rows | columns |
|---|---|---|
| players.csv | 510 | 10 |
| combine_results.csv | 510 | 18 |
| player_career_successes.csv | 510 | 9 |
| games.csv | 1002 | 7 |
| player_play.csv | 314197 | 64 |


### Combine tracking: players with tracked attempts, by drill type and position group

| drillType | WR | TE | DB | OL | DL | attempts |
|---|---|---|---|---|---|---|
| SKILL_DRILLS_DB |  |  | 122 |  |  | 1170 |
| FORTY_YARD_DASH | 94 | 34 | 102 | 95 | 93 | 793 |
| THREE_CONE_DRILL | 24 | 19 | 24 | 48 | 45 | 203 |
| SKILL_DRILLS_DL |  |  |  |  | 114 | 1007 |
| SKILL_DRILLS_TE |  | 42 |  |  |  | 438 |
| SHORT_SHUTTLE | 29 | 21 | 28 | 54 | 38 | 223 |
| SKILL_DRILLS_WR | 107 |  |  |  |  | 1529 |
| SKILL_DRILLS_OL |  |  |  | 121 |  | 910 |
| SKILL_DRILLS_LB |  |  |  |  | 4 | 37 |


### Game tracking coverage (cohort players only; PRE/POST included in the raw files)

| season | seasonType | playerPlays | players | games |
|---|---|---|---|---|
| 2023 | POST | 1558 | 31 | 13 |
| 2023 | PRE | 10207 | 150 | 49 |
| 2023 | REG | 39146 | 122 | 271 |
| 2024 | POST | 4291 | 75 | 13 |
| 2024 | PRE | 18210 | 296 | 49 |
| 2024 | REG | 86219 | 247 | 272 |
| 2025 | POST | 7196 | 115 | 13 |
| 2025 | PRE | 22375 | 420 | 49 |
| 2025 | REG | 125742 | 353 | 272 |


### Regular-season NGS outcome coverage in player_play (non-null player-plays)

| combinePosition | regPlayerPlays | separationAtPassForward | cushion | playerGetOff | peakPressureProbabilityAllowed | pressureAllowed | timeToPressure | coverageAssignment |
|---|---|---|---|---|---|---|---|---|
| DB | 66916 | 9 | 40485 | 520 | 0 | 1 | 251 | 39581 |
| DL | 52609 | 0 | 3738 | 33417 | 0 | 0 | 2704 | 29052 |
| OL | 86582 | 10 | 0 | 0 | 46493 | 48698 | 0 | 0 |
| TE | 7735 | 6781 | 6139 | 0 | 286 | 407 | 0 | 0 |
| WR | 35427 | 31035 | 33566 | 0 | 49 | 95 | 0 | 0 |

## Processing

- **combine_tracking.csv rows**: 463189
- **combine_tracking.csv plays/attempts**: 6310
- **combine_tracking.csv rows after cleaning**: 463189
- **combine_tracking.csv output**: data/processed/combine_tracking_clean.parquet
- **game_tracking_2023.csv rows**: 3665443
- **game_tracking_2023.csv plays/attempts**: 31690
- **game_tracking_2023.csv rows after cleaning**: 3665446
- **game_tracking_2023.csv output**: data/processed/game_tracking_2023_clean.parquet
- **game_tracking_2024.csv rows**: 7827944
- **game_tracking_2024.csv plays/attempts**: 40331
- **game_tracking_2024.csv rows after cleaning**: 7827945
- **game_tracking_2024.csv output**: data/processed/game_tracking_2024_clean.parquet
- **game_tracking_2025.csv rows**: 11160903
- **game_tracking_2025.csv plays/attempts**: 42544
- **game_tracking_2025.csv rows after cleaning**: 11160905
- **game_tracking_2025.csv output**: data/processed/game_tracking_2025_clean.parquet


## combine_tracking.csv

### Summary

| issue | nFlags | nPlays | description |
|---|---|---|---|
| duplicates | 0 |  | Rows repeated on (play, entity, frameId) |
| missing_xy | 0 |  | Rows with null x or y |
| frame_gaps | 0 |  | Missing frames inside an entity's trajectory |
| partial_presence | 0 |  | Entity present for fewer frames than the play/attempt spans |
| impossible_kinematics | 0 |  | Player s or a above physical limits, or negative (s, a are unsigned) |
| teleports | 2 | 2 | Frame-to-frame jump too large or inconsistent with `dis` |
| out_of_bounds | 0 |  | Position beyond the field by more than the margin (game data only) |
| player_counts | 0 |  | Plays with frames not having 11 per team (+1 ball); skipped for sparse data |
| event_issues | 0 |  | Plays with missing/multiple snaps, repeated one-off events or mis-ordered events |
| event_inconsistent | 0 |  | Frames whose rows disagree on the event tag |
| orientation | 0 |  | Player-plays where o deviates from dir by >90 deg while running |
| orientation_by_game | 0 |  | Games with systematic orientation offset (historic NGS issue) |


### teleports (2)

Frame-to-frame jump too large or inconsistent with `dis`


| draftYear | eventId | nflId | frameId | x | y | step | dis | s |
|---|---|---|---|---|---|---|---|---|
| 2025 | 20250301100700_1740864863615_000 | 58493 | 22 | 34.30 | 42.12 | 0.70 | 0.00 | 6.90 |
| 2025 | 20250301100700_1740864863616_000 | 58239 | 16 | 29.88 | 41.49 | 0.68 | 0.00 | 6.64 |


## game_tracking_2023.csv

### Summary

| issue | nFlags | nPlays | description |
|---|---|---|---|
| duplicates | 0 |  | Rows repeated on (play, entity, frameId) |
| missing_xy | 0 |  | Rows with null x or y |
| frame_gaps | 3 | 3 | Missing frames inside an entity's trajectory |
| partial_presence | 22 | 22 | Entity present for fewer frames than the play/attempt spans |
| impossible_kinematics | 14 | 4 | Player s or a above physical limits, or negative (s, a are unsigned) |
| teleports | 85 | 82 | Frame-to-frame jump too large or inconsistent with `dis` |
| out_of_bounds | 1983 | 356 | Position beyond the field by more than the margin (game data only) |
| player_counts | 0 |  | Plays with frames not having 11 per team (+1 ball); skipped for sparse data |
| event_issues | 89 | 89 | Plays with missing/multiple snaps, repeated one-off events or mis-ordered events |
| event_inconsistent | 0 |  | Frames whose rows disagree on the event tag |
| orientation | 3598 | 3462 | Player-plays where o deviates from dir by >90 deg while running |
| orientation_by_game | 17 | 17 | Games with systematic orientation offset (historic NGS issue) |


### frame_gaps (3)

Missing frames inside an entity's trajectory


| gameId | playId | nflId | frameAfterGap | nMissing |
|---|---|---|---|---|
| 2023081151 | 93 | 55922 | 55 | 1 |
| 2023081151 | 115 | 55922 | 45 | 1 |
| 2023081151 | 1836 | 55922 | 50 | 1 |


### partial_presence (22)

Entity present for fewer frames than the play/attempt spans


| gameId | playId | nflId | lo | hi | playLo | playHi |
|---|---|---|---|---|---|---|
| 2023081151 | 138 | 55922 | 11 | 106 | 1 | 117 |
| 2023081151 | 332 | 55922 | 10 | 60 | 1 | 71 |
| 2023081151 | 369 | 55922 | 10 | 90 | 1 | 101 |
| 2023081151 | 407 | 55922 | 11 | 63 | 1 | 74 |
| 2023081151 | 432 | 55922 | 11 | 74 | 1 | 84 |
| 2023081151 | 901 | 55922 | 10 | 44 | 1 | 55 |
| 2023081151 | 923 | 55922 | 10 | 86 | 1 | 96 |
| 2023081151 | 948 | 55922 | 10 | 68 | 1 | 78 |
| 2023081151 | 986 | 55922 | 11 | 69 | 1 | 80 |
| 2023081151 | 1006 | 55922 | 10 | 60 | 1 | 70 |


### impossible_kinematics (14)

Player s or a above physical limits, or negative (s, a are unsigned)


| gameId | playId | nflId | frameId | s | a |
|---|---|---|---|---|---|
| 2023081151 | 138 | 55922 | 75 | 12.05 | 0.00 |
| 2023081151 | 138 | 55922 | 76 | 12.37 | 0.00 |
| 2023081151 | 138 | 55922 | 77 | 12.58 | 0.00 |
| 2023081151 | 138 | 55922 | 78 | 12.63 | 0.00 |
| 2023081151 | 138 | 55922 | 79 | 12.57 | 0.00 |
| 2023081151 | 138 | 55922 | 80 | 12.34 | 0.00 |
| 2023091011 | 1759 | 56075 | 75 | 7.05 | 15.60 |
| 2023091011 | 614 | 56052 | 29 | 7.93 | 18.32 |
| 2023091011 | 614 | 56052 | 30 | 10.41 | 17.78 |
| 2023091011 | 614 | 56052 | 32 | 12.44 | 3.11 |


### teleports (85)

Frame-to-frame jump too large or inconsistent with `dis`


| gameId | playId | nflId | frameId | x | y | step | dis | s |
|---|---|---|---|---|---|---|---|---|
| 2023081151 | 93 | 55922 | 10 | 43.33 | 24.25 | 39.66 | 0.00 | 0.00 |
| 2023081151 | 115 | 55922 | 10 | 41.38 | 24.20 | 40.58 | 0.00 | 0.00 |
| 2023081151 | 332 | 55922 | 26 | 32.33 | 22.29 | 1.29 | 0.29 | 3.56 |
| 2023081151 | 923 | 55922 | 62 | 99.92 | 23.26 | 1.17 | 0.52 | 6.35 |
| 2023081151 | 948 | 55922 | 29 | 83.08 | 24.87 | 0.96 | 0.25 | 6.83 |
| 2023081151 | 948 | 55922 | 30 | 83.04 | 23.93 | 0.94 | 0.17 | 6.34 |
| 2023081151 | 1649 | 55922 | 11 | 56.00 | 24.62 | 0.17 | 0.93 | 0.17 |
| 2023081151 | 1836 | 55922 | 10 | 96.00 | 30.33 | 34.07 | 0.00 | 0.00 |
| 2023081152 | 1091 | 56056 | 60 | -1.80 | -1.52 | 1.71 | 1.71 | 6.35 |
| 2023081154 | 3098 | 56089 | 49 | 85.26 | 13.66 | 2.24 | 2.24 | 2.25 |


### out_of_bounds (1983)

Position beyond the field by more than the margin (game data only)


| gameId | playId | nflId | frameId | x | y |
|---|---|---|---|---|---|
| 2023102203 | 348 | 56075 | 77 | 43.38 | 56.35 |
| 2023102203 | 348 | 56075 | 78 | 43.24 | 56.64 |
| 2023102203 | 348 | 56075 | 79 | 43.13 | 56.89 |
| 2023102203 | 348 | 56075 | 80 | 43.04 | 57.14 |
| 2023102203 | 348 | 56075 | 81 | 42.94 | 57.33 |
| 2023081154 | 2379 | 56062 | 76 | 81.74 | -3.03 |
| 2023081154 | 2379 | 56062 | 77 | 81.99 | -3.21 |
| 2023081154 | 2379 | 56062 | 78 | 82.19 | -3.36 |
| 2023081154 | 2379 | 56062 | 79 | 82.43 | -3.54 |
| 2023112606 | 874 | 56042 | 67 | 25.32 | -3.30 |


### event_issues (89)

Plays with missing/multiple snaps, repeated one-off events or mis-ordered events


| gameId | playId | nSnap | snapFrame | passFrame | repeated | problem |
|---|---|---|---|---|---|---|
| 2023081151 | 1375 | 1 | 10 | 30 | True | repeated_event; |
| 2023081251 | 3860 | 1 | 10 |  | True | repeated_event; |
| 2023081252 | 3590 | 1 | 11 | 10 | False | pass_before_snap; |
| 2023081255 | 4187 | 2 | 11 |  | True | multiple_snaps;repeated_event; |
| 2023081351 | 3512 | 0 |  |  | False | no_snap; |
| 2023081352 | 2577 | 1 | 11 | 43 | True | repeated_event; |
| 2023081751 | 1872 | 1 | 10 | 38 | True | repeated_event; |
| 2023081952 | 235 | 1 | 10 | 9 | False | pass_before_snap; |
| 2023081955 | 4205 | 1 | 10 | 41 | True | repeated_event; |
| 2023081957 | 2420 | 2 | 11 | 35 | True | multiple_snaps;repeated_event; |


### orientation (3598)

Player-plays where o deviates from dir by >90 deg while running


| gameId | playId | nflId | n | medAbsDev | medDev |
|---|---|---|---|---|---|
| 2023080351 | 267 | 55976 | 11 | 104.73 | -104.73 |
| 2023080351 | 484 | 56250 | 48 | 129.04 | 16.62 |
| 2023080351 | 527 | 56250 | 36 | 94.94 | -94.94 |
| 2023080351 | 706 | 56007 | 17 | 91.86 | -53.39 |
| 2023080351 | 1544 | 55976 | 14 | 97.42 | -97.42 |
| 2023080351 | 2326 | 56250 | 27 | 121.19 | 116.75 |
| 2023080351 | 2398 | 56055 | 6 | 115.42 | -115.42 |
| 2023080351 | 2431 | 56250 | 28 | 94.91 | 5.82 |
| 2023080351 | 2579 | 56250 | 33 | 149.71 | 48.32 |
| 2023080351 | 2767 | 56007 | 24 | 136.53 | 71.05 |


### orientation_by_game (17)

Games with systematic orientation offset (historic NGS issue)


| gameId | nPlayerPlays | fracFlagged | medOffset |
|---|---|---|---|
| 2023091010 | 35 | 0.46 | -4.45 |
| 2023101508 | 80 | 0.31 | 3.89 |
| 2023101600 | 59 | 0.36 | -2.73 |
| 2023102209 | 95 | 0.27 | -0.96 |
| 2023110600 | 93 | 0.38 | -4.88 |
| 2023111208 | 132 | 0.36 | 0.32 |
| 2023121010 | 125 | 0.38 | -3.65 |
| 2023121400 | 137 | 0.28 | -2.15 |
| 2023121711 | 198 | 0.33 | -2.98 |
| 2023122301 | 112 | 0.31 | -0.38 |


## game_tracking_2024.csv

### Summary

| issue | nFlags | nPlays | description |
|---|---|---|---|
| duplicates | 0 |  | Rows repeated on (play, entity, frameId) |
| missing_xy | 0 |  | Rows with null x or y |
| frame_gaps | 3 | 2 | Missing frames inside an entity's trajectory |
| partial_presence | 5 | 2 | Entity present for fewer frames than the play/attempt spans |
| impossible_kinematics | 8 | 2 | Player s or a above physical limits, or negative (s, a are unsigned) |
| teleports | 164 | 151 | Frame-to-frame jump too large or inconsistent with `dis` |
| out_of_bounds | 4046 | 747 | Position beyond the field by more than the margin (game data only) |
| player_counts | 0 |  | Plays with frames not having 11 per team (+1 ball); skipped for sparse data |
| event_issues | 119 | 119 | Plays with missing/multiple snaps, repeated one-off events or mis-ordered events |
| event_inconsistent | 0 |  | Frames whose rows disagree on the event tag |
| orientation | 6563 | 6026 | Player-plays where o deviates from dir by >90 deg while running |
| orientation_by_game | 9 | 9 | Games with systematic orientation offset (historic NGS issue) |


### frame_gaps (3)

Missing frames inside an entity's trajectory


| gameId | playId | nflId | frameAfterGap | nMissing |
|---|---|---|---|---|
| 2024081059 | 4111 | 56625 | 21 | 7 |
| 2024081059 | 4111 | 57637 | 21 | 7 |
| 2024100605 | 132 | 55911 | 76 | 1 |


### partial_presence (5)

Entity present for fewer frames than the play/attempt spans


| gameId | playId | nflId | lo | hi | playLo | playHi |
|---|---|---|---|---|---|---|
| 2024081757 | 2551 | 57283 | 1 | 48 | 1 | 49 |
| 2024112402 | 2297 | 57163 | 1 | 61 | 1 | 62 |
| 2024112402 | 2297 | 57128 | 1 | 61 | 1 | 62 |
| 2024112402 | 2297 | 55875 | 1 | 61 | 1 | 62 |
| 2024112402 | 2297 | 57199 | 1 | 61 | 1 | 62 |


### impossible_kinematics (8)

Player s or a above physical limits, or negative (s, a are unsigned)


| gameId | playId | nflId | frameId | s | a |
|---|---|---|---|---|---|
| 2024102006 | 3650 | 57145 | 13 | 13.87 | 22.02 |
| 2024102006 | 3650 | 57145 | 14 | 13.43 | 17.00 |
| 2024102006 | 3650 | 57145 | 15 | 13.11 | 12.35 |
| 2024102006 | 3650 | 57145 | 18 | 1.26 | 16.54 |
| 2024102006 | 3650 | 57145 | 19 | 4.22 | 19.84 |
| 2024102006 | 3650 | 57145 | 20 | 6.15 | 18.47 |
| 2024102006 | 3650 | 57145 | 21 | 7.57 | 15.49 |
| 2024080951 | 957 | 56090 | 38 | 1.72 | 15.22 |


### teleports (164)

Frame-to-frame jump too large or inconsistent with `dis`


| gameId | playId | nflId | frameId | x | y | step | dis | s |
|---|---|---|---|---|---|---|---|---|
| 2024080951 | 957 | 56090 | 31 | 109.14 | 19.15 | 1.74 | 1.74 | 6.30 |
| 2024080951 | 957 | 56090 | 32 | 110.30 | 17.60 | 1.94 | 1.94 | 9.71 |
| 2024080951 | 957 | 56090 | 33 | 111.28 | 16.32 | 1.61 | 1.61 | 11.56 |
| 2024080951 | 3273 | 57156 | 44 | 76.28 | 28.31 | 1.73 | 1.72 | 0.00 |
| 2024080951 | 3722 | 57156 | 50 | 49.19 | 21.06 | 1.81 | 1.81 | 1.47 |
| 2024080951 | 3722 | 57156 | 74 | 51.92 | 12.96 | 2.45 | 2.45 | 0.08 |
| 2024080951 | 3747 | 57156 | 50 | 60.53 | 23.34 | 2.49 | 2.49 | 4.23 |
| 2024080951 | 3747 | 57156 | 51 | 61.93 | 24.25 | 1.67 | 1.67 | 5.05 |
| 2024080951 | 3769 | 57156 | 41 | 68.39 | 31.78 | 1.82 | 1.82 | 0.33 |
| 2024080952 | 2373 | 56361 | 66 | 77.09 | -1.16 | 1.52 | 1.52 | 7.73 |


### out_of_bounds (4046)

Position beyond the field by more than the margin (game data only)


| gameId | playId | nflId | frameId | x | y |
|---|---|---|---|---|---|
| 2024102800 | 3636 | 57191 | 82 | 19.81 | -3.19 |
| 2024102800 | 3636 | 57191 | 83 | 19.40 | -3.50 |
| 2024102800 | 3636 | 57191 | 84 | 19.02 | -3.80 |
| 2024102800 | 3636 | 57191 | 85 | 18.67 | -4.07 |
| 2024102800 | 3636 | 57191 | 86 | 18.35 | -4.32 |
| 2024102800 | 3636 | 57191 | 87 | 18.06 | -4.56 |
| 2024102800 | 3636 | 57191 | 88 | 17.80 | -4.77 |
| 2024102800 | 3636 | 57191 | 89 | 17.59 | -4.95 |
| 2024102800 | 3636 | 57191 | 90 | 17.41 | -5.08 |
| 2024102800 | 3636 | 57191 | 91 | 17.27 | -5.19 |


### event_issues (119)

Plays with missing/multiple snaps, repeated one-off events or mis-ordered events


| gameId | playId | nSnap | snapFrame | passFrame | repeated | problem |
|---|---|---|---|---|---|---|
| 2024080851 | 1895 | 2 | 11 | 61 | True | multiple_snaps;repeated_event; |
| 2024081051 | 308 | 2 | 10 |  | True | multiple_snaps;repeated_event; |
| 2024081054 | 2077 | 1 | 10 | 35 | True | repeated_event; |
| 2024081751 | 3591 | 1 | 10 | 26 | True | repeated_event; |
| 2024081753 | 3749 | 1 | 10 | 29 | True | repeated_event; |
| 2024081759 | 3496 | 1 | 11 | 75 | True | repeated_event; |
| 2024081851 | 2454 | 1 | 11 |  | True | repeated_event; |
| 2024082251 | 3087 | 1 | 11 | 86 | True | repeated_event; |
| 2024082252 | 3506 | 1 | 11 | 23 | True | repeated_event; |
| 2024082353 | 3192 | 1 | 10 | 46 | True | repeated_event; |


### orientation (6563)

Player-plays where o deviates from dir by >90 deg while running


| gameId | playId | nflId | n | medAbsDev | medDev |
|---|---|---|---|---|---|
| 2024080151 | 110 | 57180 | 6 | 95.81 | 95.81 |
| 2024080151 | 319 | 57180 | 10 | 126.63 | 126.63 |
| 2024080151 | 796 | 56095 | 5 | 130.28 | -130.28 |
| 2024080151 | 864 | 57180 | 15 | 107.76 | 107.76 |
| 2024080151 | 889 | 57265 | 9 | 109.91 | -109.91 |
| 2024080851 | 458 | 57278 | 41 | 95.33 | 73.01 |
| 2024080851 | 506 | 55982 | 8 | 91.73 | -91.73 |
| 2024080851 | 506 | 57721 | 15 | 111.93 | 111.93 |
| 2024080851 | 948 | 57721 | 51 | 117.85 | 117.85 |
| 2024080851 | 1094 | 56117 | 14 | 99.06 | 99.06 |


### orientation_by_game (9)

Games with systematic orientation offset (historic NGS issue)


| gameId | nPlayerPlays | fracFlagged | medOffset |
|---|---|---|---|
| 2024080852 | 235 | 0.28 | 0.32 |
| 2024091500 | 76 | 0.29 | -0.15 |
| 2024100602 | 122 | 0.27 | 4.25 |
| 2024101300 | 272 | 0.32 | -25.03 |
| 2024120107 | 237 | 0.25 | 4.62 |
| 2024120109 | 176 | 0.35 | 12.08 |
| 2024121505 | 110 | 0.33 | 4.91 |
| 2025011101 | 136 | 0.41 | -4.00 |
| 2025011901 | 134 | 0.26 | 0.23 |


## game_tracking_2025.csv

### Summary

| issue | nFlags | nPlays | description |
|---|---|---|---|
| duplicates | 0 |  | Rows repeated on (play, entity, frameId) |
| missing_xy | 0 |  | Rows with null x or y |
| frame_gaps | 2 | 2 | Missing frames inside an entity's trajectory |
| partial_presence | 6 | 4 | Entity present for fewer frames than the play/attempt spans |
| impossible_kinematics | 174 | 5 | Player s or a above physical limits, or negative (s, a are unsigned) |
| teleports | 156 | 136 | Frame-to-frame jump too large or inconsistent with `dis` |
| out_of_bounds | 5203 | 985 | Position beyond the field by more than the margin (game data only) |
| player_counts | 0 |  | Plays with frames not having 11 per team (+1 ball); skipped for sparse data |
| event_issues | 99 | 99 | Plays with missing/multiple snaps, repeated one-off events or mis-ordered events |
| event_inconsistent | 0 |  | Frames whose rows disagree on the event tag |
| orientation | 9570 | 8420 | Player-plays where o deviates from dir by >90 deg while running |
| orientation_by_game | 0 |  | Games with systematic orientation offset (historic NGS issue) |


### frame_gaps (2)

Missing frames inside an entity's trajectory


| gameId | playId | nflId | frameAfterGap | nMissing |
|---|---|---|---|---|
| 2025080753 | 3834 | 56645 | 83 | 1 |
| 2025111000 | 3701 | 57179 | 75 | 1 |


### partial_presence (6)

Entity present for fewer frames than the play/attempt spans


| gameId | playId | nflId | lo | hi | playLo | playHi |
|---|---|---|---|---|---|---|
| 2025080753 | 3834 | 56645 | 1 | 84 | 1 | 85 |
| 2025101205 | 1289 | 57155 | 1 | 90 | 1 | 92 |
| 2025102001 | 406 | 55867 | 1 | 94 | 1 | 95 |
| 2025102001 | 406 | 55973 | 1 | 94 | 1 | 95 |
| 2025111000 | 3701 | 57232 | 1 | 73 | 1 | 75 |
| 2025111000 | 3701 | 55930 | 1 | 73 | 1 | 75 |


### impossible_kinematics (174)

Player s or a above physical limits, or negative (s, a are unsigned)


| gameId | playId | nflId | frameId | s | a |
|---|---|---|---|---|---|
| 2025101205 | 1289 | 57155 | 2 | 0.05 | -0.07 |
| 2025101205 | 1289 | 57155 | 4 | 0.09 | -0.51 |
| 2025101205 | 1289 | 57155 | 6 | 0.06 | -0.34 |
| 2025101205 | 1289 | 57155 | 8 | 0.04 | -0.41 |
| 2025101205 | 1289 | 57155 | 10 | 0.04 | -0.11 |
| 2025101205 | 1289 | 57155 | 45 | 12.06 | 0.00 |
| 2025101205 | 1289 | 57155 | 46 | 12.49 | 0.00 |
| 2025101205 | 1289 | 57155 | 47 | 12.82 | 0.00 |
| 2025101205 | 1289 | 57155 | 48 | 13.33 | 0.00 |
| 2025101205 | 1289 | 57155 | 49 | 13.48 | 0.00 |


### teleports (156)

Frame-to-frame jump too large or inconsistent with `dis`


| gameId | playId | nflId | frameId | x | y | step | dis | s |
|---|---|---|---|---|---|---|---|---|
| 2025073151 | 508 | 58363 | 68 | 73.48 | -1.50 | 1.73 | 1.73 | 7.96 |
| 2025080752 | 3393 | 57305 | 51 | 116.70 | 55.21 | 2.07 | 2.07 | 6.79 |
| 2025080851 | 1722 | 58998 | 55 | 73.62 | -1.83 | 2.03 | 2.03 | 6.30 |
| 2025080951 | 4219 | 58424 | 71 | 96.68 | 55.14 | 1.97 | 1.97 | 3.40 |
| 2025080953 | 2618 | 58772 | 96 | 20.60 | -1.52 | 1.71 | 1.70 | 2.92 |
| 2025080958 | 935 | 56020 | 61 | 43.20 | -1.83 | 2.01 | 2.01 | 4.72 |
| 2025080958 | 1847 | 58279 | 54 | 101.65 | 54.96 | 1.80 | 1.80 | 4.59 |
| 2025081051 | 277 | 57304 | 56 | 66.50 | -1.22 | 1.50 | 1.50 | 6.97 |
| 2025081051 | 752 | 57265 | 75 | 106.26 | 54.15 | 2.00 | 2.00 | 7.06 |
| 2025081051 | 752 | 57361 | 75 | 103.54 | 55.15 | 1.98 | 1.99 | 4.61 |


### out_of_bounds (5203)

Position beyond the field by more than the margin (game data only)


| gameId | playId | nflId | frameId | x | y |
|---|---|---|---|---|---|
| 2025111601 | 2139 | 58323 | 79 | 64.70 | 56.45 |
| 2025110905 | 4035 | 58229 | 66 | -3.09 | 10.81 |
| 2025110905 | 4035 | 58229 | 67 | -3.42 | 10.82 |
| 2025110905 | 4035 | 58229 | 68 | -3.73 | 10.83 |
| 2025110905 | 4035 | 58229 | 69 | -4.02 | 10.82 |
| 2025110905 | 4035 | 58229 | 70 | -4.25 | 10.82 |
| 2025110905 | 4035 | 58229 | 71 | -4.42 | 10.80 |
| 2025110905 | 4035 | 58229 | 72 | -4.55 | 10.80 |
| 2025112700 | 1000 | 55910 | 65 | 114.74 | -3.14 |
| 2025112700 | 1000 | 55910 | 66 | 114.93 | -3.39 |


### event_issues (99)

Plays with missing/multiple snaps, repeated one-off events or mis-ordered events


| gameId | playId | nSnap | snapFrame | passFrame | repeated | problem |
|---|---|---|---|---|---|---|
| 2025080853 | 3293 | 0 |  |  | False | no_snap; |
| 2025080952 | 514 | 1 | 10 |  | True | repeated_event; |
| 2025080953 | 4282 | 1 | 11 | 56 | True | repeated_event; |
| 2025080955 | 2746 | 0 |  |  | False | no_snap; |
| 2025081552 | 2597 | 1 | 10 |  | True | repeated_event; |
| 2025081651 | 911 | 1 | 11 |  | True | repeated_event; |
| 2025081652 | 483 | 0 |  | 46 | False | no_snap; |
| 2025081652 | 508 | 0 |  | 34 | False | no_snap; |
| 2025081652 | 971 | 1 | 11 | 31 | True | repeated_event; |
| 2025081652 | 3069 | 1 | 10 | 37 | True | repeated_event; |


### orientation (9570)

Player-plays where o deviates from dir by >90 deg while running


| gameId | playId | nflId | n | medAbsDev | medDev |
|---|---|---|---|---|---|
| 2025073151 | 2347 | 58330 | 6 | 92.70 | 92.70 |
| 2025073151 | 3118 | 58401 | 26 | 110.11 | 99.60 |
| 2025073151 | 3485 | 57796 | 8 | 124.73 | 124.73 |
| 2025073151 | 3625 | 58330 | 14 | 113.46 | 113.46 |
| 2025080751 | 122 | 58229 | 55 | 91.76 | 80.17 |
| 2025080751 | 152 | 58229 | 35 | 175.21 | 99.36 |
| 2025080751 | 188 | 58229 | 33 | 163.96 | -61.42 |
| 2025080751 | 210 | 55971 | 6 | 145.89 | 145.89 |
| 2025080751 | 210 | 58229 | 18 | 123.03 | 123.03 |
| 2025080751 | 235 | 58229 | 41 | 119.41 | 74.98 |
