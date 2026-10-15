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

