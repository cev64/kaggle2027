# External data (nflverse)

External context for the Big Data Bowl tracking data, pulled with `nflreadpy` 0.1.5 and
cached as parquet under `data/external/` (gitignored). Checked 2026-10-15 for seasons 2022-2025.

```bash
python scripts/fetch_external.py                      # all datasets, 2022-2025 (~45 s, 77 MB)
python scripts/fetch_external.py --seasons 2025 --datasets pbp participation --refresh
```

```python
from bdb import external, keys
pbp = external.load("pbp", [2024, 2025])             # polars; fetched + cached on first use
plays = keys.merge_plays(bdb_plays)                   # gameId/playId -> pbp + participation + FTN
players = keys.merge_players(bdb_players)             # nflId -> gsis_id, bio, draft
onfield = keys.participation_long([2024])             # one row per player per play, with nflId
games = keys.merge_games(bdb_games)                   # gameId -> roof, surface, weather, lines
```

Layout: seasonal datasets live at `data/external/<name>/<season>.parquet`; `players`, `combine`
and `draft_picks` are single files. `external.load` concatenates seasons with
`diagonal_relaxed`, so schema drift between seasons becomes nulls, not errors.

## Join keys (verified)

| BDB | nflverse | Notes |
|---|---|---|
| `gameId` (int, `YYYYMMDDNN`) | `old_game_id` (str) in `schedules`, `pbp`, `participation` | Cast to Int64. Unique per game; all 1,139 schedule games match pbp exactly. |
| (via schedules) | `game_id` (`2024_01_BAL_KC`) | Used by FTN (`nflverse_game_id`), PFR, snap counts. `keys.game_keys()` is the crosswalk. |
| `playId` | `play_id` in `pbp`, `participation`, FTN (`nflverse_play_id`) | GSIS play id. `(old_game_id, play_id)` is unique in pbp and participation; `(game_id, play_id)` is unique in FTN. All FTN rows match pbp. |
| `nflId` | `players.nfl_id` (string; `nfl_id_int` added on cache) = `rosters.gsis_it_id` | See below. |
| season | `keys.season_from_game_id` | Jan-Mar games belong to the previous season (`2025020900` is the 2024 Super Bowl). |

**`nflId` mapping evidence.** In 2022 participation, `players_on_play` holds NGS player ids for
45,859 of 50,150 plays (the NGS feed itself), while `offense_players`/`defense_players` hold
GSIS ids for the same plays. Mapping those GSIS ids through `players.gsis_id -> nfl_id`
gives an exact match: 100% of the mapped ids are in the NGS list, and 99.998% of NGS ids are
covered. `nfl_id` has no duplicates and is 1:1 with `gsis_id`. `rosters.gsis_it_id` equals `nfl_id`
for 100% of 2022-2025 roster rows. Spot checks: Mahomes 44822, Kelce 40011, J. Jefferson 52430.
Every player on the field in 2022-2025 participation has an `nfl_id`, `esb_id`, height and
birth date in `players`. (12,930 of 24,854 `players` rows lack `nfl_id`; these are historical players.)

Everything else in nflverse keys players by `gsis_id` (`00-00xxxxx`): pbp `*_player_id`,
participation lists, rosters, depth charts, injuries, NGS `player_gsis_id`, draft picks.
PFR tables and snap counts key on `pfr_player_id`, which is `players.pfr_id` (99.6% of snap-count players match). Combine has
`pfr_id` only. Use `keys.player_crosswalk()` to move between them.

## Row counts

| dataset | 2022 | 2023 | 2024 | 2025 | key / grain |
|---|---:|---:|---:|---:|---|
| pbp (REG+POST) | 49,434 | 49,665 | 49,492 | 48,771 | game_id, play_id |
| participation | 50,150 | 46,168 | 45,919 | 45,184 | old_game_id, play_id |
| ftn_charting | 41,643 | 48,225 | 48,031 | 47,316 | game_id, play_id |
| schedules | 284 | 285 | 285 | 285 | game_id |
| rosters / rosters_weekly | 3,134 / 46,163 | 3,090 / 45,655 | 3,216 / 46,579 | 3,137 / 46,849 | gsis_id (x week) |
| snap_counts | 26,381 | 26,540 | 26,615 | 26,613 | game_id, pfr_player_id |
| injuries | 5,682 | 5,599 | 6,215 | 6,068 | week, gsis_id |
| depth_charts | 37,780 | 37,327 | 37,312 | **554,215** | see quirks |
| ngs passing / rushing / receiving | 603 / 617 / 1,466 | 620 / 623 / 1,473 | 614 / 601 / 1,435 | 605 / 648 / 1,402 | week, player_gsis_id (week 0 = season) |
| pfr pass / rush / rec / def (weekly) | 685 / 2,389 / 4,547 / 7,878 | 700 / 2,380 / 4,594 / 7,902 | 697 / 2,359 / 4,453 / 7,992 | 684 / 2,355 / 4,533 / 7,926 | game_id, pfr_player_id |
| players / combine / draft_picks | 24,854 / 8,968 / 12,927 (all years) | | | | gsis_id / pfr_id / gsis_id |

Disk: 77 MB total, of which pbp is 52 MB and participation 9 MB.

## Coverage on scrimmage plays

Scrimmage here means every pbp play with a play_type; dropbacks are pbp `pass_attempt == 1`.
Values are the share of rows that are non-null.

| | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| in participation | 1.000 | 0.957 | 0.956 | 0.955 |
| offense/defense personnel | 0.797 | 0.957 | 0.956 | 0.955 |
| defenders_in_box | 0.777 | 0.957 | 0.955 | 0.955 |
| number_of_pass_rushers | 0.446 | 0.931 | 0.955 | 0.955 |
| in FTN charting | 0.868 | 1.000 | 1.000 | 1.000 |
| coverage type / man-zone (dropbacks) | 0.928 | 0.994 | 0.995 | 0.995 |
| route (dropbacks) | 0.888 | 0.898 | 0.897 | 0.903 |
| time_to_throw (dropbacks) | 0.926 | 0.924 | 0.925 | 0.926 |
| ngs_air_yards (dropbacks) | 0.890 | **0** | **0** | **0** |

Postseason coverage matches regular season in every year.

## Quirks and gotchas

- **Participation does exist for 2024 and 2025.** The fear that it stopped after 2023 is wrong
  for nflverse as of Oct 2026. But from 2023 on the source and vocabulary changed (it looks FTN-sourced):
  - Personnel strings are full position counts (`1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR`) instead of
    2022's `1 RB, 1 TE, 3 WR`. Defense uses `CB/DE/DT/FS/SS/ILB/OLB` counts, not `4 DL, 2 LB, 5 DB`.
    Parse the RB/TE/WR counts yourself to compare across years.
  - Formation values are `SHOTGUN / UNDER CENTER / PISTOL` (2023+). 2022 used `SHOTGUN / SINGLEBACK / EMPTY / I_FORM / PISTOL / JUMBO / WILDCAT`.
  - Coverage values add `COMBO`, `COVER_9` and `BLOWN` from 2023 on. Man/zone is `MAN_COVERAGE` / `ZONE_COVERAGE`.
  - Route values change too: `HITCH/CURL`, `QUICK OUT`, `IN/DIG`, `SCREEN` (2023+) versus `HITCH`, `OUT`, `IN`, `FLAT` (2022).
    There is one route per play (the targeted receiver's), never one per player.
  - `ngs_air_yards` is gone after 2022.
  - The `offense_/defense_ positions/names/numbers` lists exist only from 2023 on. They are
    positionally aligned with `offense_players`/`defense_players`; 97% of names match `players.display_name`
    exactly, and the rest are name variants such as Jr.
  - `play_id` is a float in the 2025 file. The cache casts it to Int64, and empty strings are cast to null.
  - 2023+ participation omits `no_play` (penalty) rows, which is the whole 4.4% gap. 2022 includes
    them, plus 716 empty timing rows that aren't in pbp.
  - On kickoffs, "offense" is the possession (receiving) team.
- **2022 participation `players_on_play`** mixes NGS `nflId`s (most rows) with GSIS ids. Use
  `offense_players`/`defense_players` (always GSIS) and `keys.participation_long`.
- **FTN 2022** covers about 87% of plays, with a few plays missing in every week. 2023+ coverage is complete.
- **pbp**: no duplicate keys; `epa`/`wpa`/`wp` are never null on run or pass plays. `cp`/`cpoe` are null on
  about 5% of non-sack pass attempts (throwaways, spikes, missing air yards), and `xpass` on about 1%. `play_id` is cast to Int64.
- **schedules**: `temp`/`wind` are null for 44% of games (domes and closed roofs); `surface` is `""` for 44 games.
  Weather text is also available in pbp `weather`.
- **depth_charts 2025** uses a new schema: daily snapshots keyed by `dt` timestamp, with `pos_abb`/`pos_rank`
  and no `week`. 2022-2024 are weekly with `depth_team`. Filter 2025 to the snapshot just before each game.
- **combine** includes the 2026 class. Drill fields are often null: bench 40%, cone 43%, shuttle 42%, forty 10%.
- `players.height` is in inches and `weight` in lbs. `birth_date` is an ISO string. `draft_*` is null for undrafted players (about 30% of on-field players).

## Data dictionary (most useful for tracking work)

**pbp** (`keys.PBP_COLS`; 370+ columns available)
- Situation: `posteam`, `defteam`, `qtr`, `down`, `ydstogo`, `yardline_100` (distance to the opponent end zone), `game_seconds_remaining`, `score_differential`, `wp`, `vegas_wp`.
- Play: `play_type`, `shotgun`, `no_huddle`, `qb_dropback`, `qb_scramble`, `pass_location`, `pass_length`, `air_yards`, `yards_after_catch`, `yards_gained`, `penalty`.
- Value: `epa`, `wpa`, `success`, `xyac_epa`, `xyac_mean_yardage` (expected YAC).
- Models: `xpass` (pre-snap pass probability), `pass_oe`, `cp` (completion probability), `cpoe`.
- Players (gsis): `passer_player_id`, `receiver_player_id`, `rusher_player_id`, plus tacklers, pass defenders and others.

**participation** (`keys.PARTICIPATION_COLS`)
- `offense_formation`, `offense_personnel`, `defense_personnel`, `defenders_in_box`, `number_of_pass_rushers`.
- `defense_coverage_type`, `defense_man_zone_type`, `route` (targeted route), `was_pressure`, `time_to_throw`.
- `offense_players` / `defense_players` (gsis lists); from 2023 also `*_positions`, `*_names`, `*_numbers`.

**ftn_charting** (`keys.FTN_COLS`)
- Pre-snap and design: `starting_hash`, `qb_location` (S/U/P, plus a few `0`), `n_offense_backfield`, `n_defense_box`, `is_motion`, `is_play_action`, `is_screen_pass`, `is_rpo`, `is_trick_play`, `is_no_huddle`.
- Pass rush: `n_blitzers`, `n_pass_rushers`, `is_qb_out_of_pocket`, `is_qb_fault_sack`.
- Throw and catch: `read_thrown` (which read the QB threw to), `is_throw_away`, `is_catchable_ball`, `is_contested_ball`, `is_drop`, `is_created_reception`, `is_interception_worthy`.

**schedules** (`keys.SCHEDULE_COLS`)
- `roof`, `surface`, `temp`, `wind`, `stadium`, `spread_line` (home perspective), `total_line`, `home_rest`, `away_rest`, coaches, `referee`.

**players** (`keys.PLAYER_COLS`)
- IDs: `gsis_id`, `nfl_id` (= BDB `nflId`), `esb_id`, `smart_id`, `pff_id`, `pfr_id`, `espn_id`.
- Bio: `position`, `position_group`, `ngs_position`, `height`, `weight`, `birth_date`, `college_name`, `rookie_season`, `years_of_experience`.
- Draft: `draft_year`, `draft_round`, `draft_pick`, `draft_team`.

**ngs_*** (player-week; week 0 is the season total)
- Passing: `avg_time_to_throw`, `aggressiveness`, `avg_air_yards_to_sticks`, `completion_percentage_above_expectation`.
- Receiving: `avg_cushion`, `avg_separation`, `avg_yac_above_expectation`.
- Rushing: `efficiency`, `percent_attempts_gte_eight_defenders`, `avg_time_to_los`, `rush_yards_over_expected`.

**pfr_*** (player-game): drops, bad throws, pressures, blitzes, hurries, missed tackles, plus coverage stats allowed (`def_targets`, `def_completions_allowed`, `def_yards_allowed`, `def_passer_rating_allowed`, `def_adot`).

**Others**: `snap_counts` (offense/defense/ST snap %), `injuries` (report and practice status by week),
`rosters_weekly` (status, jersey, depth chart position by week), `combine` (forty, vertical, bench,
broad, cone, shuttle), and `draft_picks` (career AV).
