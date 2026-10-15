# NFL Big Data Bowl 2027: Competition Brief

*Compiled 2026-10-15. Sources are cited inline. Legend:*
- **[VERIFIED]**: read directly from the Kaggle competition pages, retrieved through Kaggle's page-content API (the HTML pages are JS-rendered), or from an official NFL source.
- **[PRESS]**: from NFL/AWS press coverage, not from the Kaggle rules.
- **[PRIOR YEAR]**: carried over from BDB 2023–2026 pages. Not confirmed for 2027.
- **[INFERENCE]**: my own reading or recommendation.
- **[UNVERIFIED]**: something I could not confirm.

Primary URLs:
- Overview: https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/overview
- Evaluation: https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/overview/evaluation
- Data: https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data
- Rules: https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/rules
- NFL Football Ops: https://operations.nfl.com/gameday/analytics/big-data-bowl and https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl
- NFL release (Oct 7, 2026): https://media.nfl.com/news-and-releases/consumer-products/national-football-league-announces-ninth-annual-big-data-bowl-po
- SVG coverage: https://www.sportsvideo.org/2026/10/08/nfl-announces-ninth-annual-big-data-bowl-powered-by-aws/

---

## 1. Competition facts

### 1.1 Theme / prompt [VERIFIED]
Kaggle brief description: *"Uncover non-obvious linkages between Combine sensor tracking and regular-season NFL game performance."*

Abstract (verbatim):
> "You are challenged to build an analytical framework that uncovers non-obvious, actionable linkages between 10 Hz Combine sensor tracking data and future regular-season NFL game performance. Your submission should look beyond conventional scouting wisdom and identifying novel signals that help coaches and evaluators identify hidden talent, avoid costly draft busts, and design position-specific player development protocols."

Description, **"2027 Theme: Combine to NFL Performance"** (key passages, verbatim):
> "The 2027 Big Data Bowl provides player tracking data from the NFL Scouting Combine alongside regular-season NFL game tracking data. The goal of this year's competition is to better understand how player movement during Combine drills translates to performance on the field during the NFL regular season."

> "participants are tasked with analyzing prospect movement in a way that is actionable and accessible to coaches, scouts, and front offices. This could include creating a new metric or comparing movement traits across position groups. As is often the case in Big Data Bowl analytics competitions, participants are encouraged to focus on one small aspect of movement, a specific position group, or an individual drill, rather than attempting to evaluate every prospect across all drills."

Examples given by the host ("not exhaustive"):
- **Route running and coverage:** "Measuring change of direction, deceleration into breaks, and acceleration out of cuts for receivers and defensive backs compared to in-game separation"
- **Trench play:** "Evaluating first-step quickness, lateral agility, and burst for offensive and defensive linemen during agility drills compared to pass-rush or run-blocking performance"
- **Drill translation:** "Comparing sensor-based movement metrics with traditional drill times to identify where tracking data adds context to prospect evaluation"
- **Position-specific mechanics:** "Analyzing body control, acceleration profiles, or pursuit angles during position drills and their relationship to NFL rookie performance"

[PRESS] The NFL lists five focus position groups: **WR, TE, DB, OL, DL**. NFL quotes: Mike Lopez (Senior Director, Football Data & Analytics) says the competition will "evaluate current and former NFL Draft prospects", and Ari Entin (AWS) says it will "bridge the college and pro game for the first time."

### 1.2 Tracks [VERIFIED in rules prize section; names differ by source]
- **Open Track**: Kaggle rules §1.5. The NFL describes it as "for all individuals or groups outside of college or university."
- **University Track**: Kaggle rules name. The NFL and press call it the "Undergraduate/Graduate Track": "Only for groups or individuals made up entirely of undergraduate or graduate students. Verification may be required."
- **No coaching, broadcast, or prediction track in 2027.** 2026 ran a separate Prediction competition plus University and Broadcast Visualization tracks. 2023–2025 had Metric, Coaching, and Undergraduate tracks. The Kaggle "tracks-and-awards" and "judges" pages for 2027 contain only placeholder text.
- [UNVERIFIED] How track selection works in the Writeup. In 2026 you had to "select a Track for your Writeup in order to submit" [PRIOR YEAR]. I also could not confirm whether students may choose the Open Track instead.

### 1.3 Timeline [VERIFIED, Kaggle Timeline page]
| Date | Event |
|---|---|
| **Oct 6, 2026** | Start date. The NFL release says sign-ups opened Oct 7. |
| **Jan 6, 2027, 11:59 PM UTC** | **Final Submission Deadline.** Per the API metadata, the team merger and new-entrant deadlines are also Jan 6, 2027, 23:59:59 UTC. |
| Jan 7 – Jan 25, 2027 | Judging period ("subject to change based on the number of submissions received") |
| **Jan 26, 2027** | Anticipated results announcement |
| Late Feb / early Mar 2027 | Finalists present at the 2027 NFL Scouting Combine (Indianapolis). [UNVERIFIED] exact date. In prior years it was the last week of February. |

Other events [PRESS]: AWS/NFL student regional events in Rio (Sep 23), London (Oct 9), Paris (Oct 21), New York (Nov 10), and Mexico City (Nov 18–20). The NFL Big Data Bowl Mentorship Program is also returning.

### 1.4 Prizes [VERIFIED, Rules §1.5]: total $100,000
- **Open Track ($45,000):** 3 Finalists × $10,000, plus 3 Runners-Up × $5,000
- **University Track ($45,000):** 3 Finalists × $10,000, plus 3 Runners-Up × $5,000
- **Grand Prize ($10,000):** 1 Combine Presentation Winner. "Each of the finalists from the Open and University tracks will receive an invitation to present at the 2027 NFL Scouting Combine, where one team will win an additional $10,000 Grand Prize. Submissions should not be made directly to the Grand Prize track."
- Team prizes are split evenly unless the team unanimously agrees otherwise (Rules §3.9.f).

### 1.5 Teams and submission count [VERIFIED]
- **Max team size: 5.** This is up from 4 in 2023–2026.
- Hackathon format: "each Team may submit one (1) Submission only." Each member needs their own Kaggle account, and you may join only one team.
- `requiresIdentityVerification: true` in the competition metadata, so you will likely need Kaggle identity verification.

### 1.6 Submission format [VERIFIED unless marked]
From the 2027 "Submission Requirements" page (verbatim):
> "**Word and figure limits:** Submissions should contain no more than 2,000 words and fewer than 10 tables or figures. Submissions over these limits may be subject to penalty."
> "**Code reproducibility:** Participants are encouraged to show statistical code if it helps readers better understand their analyses, but most code should be placed in an Appendix or linked notebook. Any writeup that does not use the player tracking data will not be scored."

**Pass/fail gates** from the 2027 Evaluation page:
| Requirement | Passing standard |
|---|---|
| 1. Combine-to-NFL Linkage | "Writeup explicitly links Combine sensor tracking data to regular-season NFL game performance." |
| 2. Writeup Completeness | "Features readable markdown, embedded visuals, and an attached **Public Kaggle Notebook**." |

- The format is a **Kaggle Writeup** (competition type is a hackathon with file-based submissions) with an **attached public Kaggle Notebook**.
- [PRIOR YEAR 2026, likely the same] A cover image in the Media Gallery was required. Draft or un-submitted Writeups at the deadline were not judged. A private resource attached to a public Writeup is auto-published after the deadline. Content is Markdown.
- **Video:** the 2027 pages do not mention video. In 2026, video (≤3 min, YouTube) was required only for the Broadcast track. [INFERENCE] Video is optional now. Animations (GIF/MP4) inside the Writeup or Media Gallery are allowed in practice, but **whether a GIF counts toward the <10 figures is not stated**.

### 1.7 Judging [VERIFIED, 2027 Evaluation page]
"All Writeups are evaluated on the following four components, where 0 is the lowest and 10 is the highest."

| Component | Weight | Questions (verbatim) |
|---|---|---|
| **Football Score** | **30%** | Would NFL teams (or the league office) be able to use these results on a week-to-week basis? Does the analysis account for variables that make football data complex? Are the ideas unique? |
| **Data Science Score** | **30%** | Is the work correct? Are claims backed up by data? Are the statistical models appropriate given the data? Are the analytical applications innovative? |
| **Writeup Score** | **20%** | Is the Writeup well-written? Is the Writeup easy to follow? Is the motivation (metric, player evaluation, etc) clearly defined? |
| **Data Visualization Score** | **20%** | Are the charts and tables provided accessible? Are the charts and tables accurate? Are the charts and tables innovative? |

Judges: "analytics staffers working for the 32 NFL teams or player tracking vendors that work with NFL teams. Scores will be summarized using a weighted average." [PRESS] The process includes "multiple rounds of evaluation by NFL analysts and football operations experts." The Combine panel has included Cynthia Frelund, former participants, AWS data scientists, and Najee Goode (NFL Ops page).

These weights are identical to 2025 and 2026.

### 1.8 Eligibility [VERIFIED, Rules]
- You must have a registered Kaggle account and be at least 18 or the age of majority. Minors may participate but cannot win prizes. Residents of Crimea, DNR, LNR, Cuba, Iran, North Korea, and sanctioned persons are excluded.
- Employees and contractors of the NFL, Kaggle, and their affiliates may participate but **cannot win prizes**.
- The University Track requires an all-student team, and verification may be required.

---

## 2. Data [VERIFIED: 2027 Kaggle data-description page]

**Overview:** "high-frequency optical tracking, standardized physical testing results, and multi-season NFL game performance metrics for **510 rookie prospects** drafted or signed between **2023 and 2025**." The data is **9 CSV files, about 2.2 GiB.** The tracking frame rate is **10 Hz** in both Combine and game data.

Note: Kaggle says "optical tracking", while the NFL release says "Zebra tracking data worn by prospects" (RFID). These two descriptions are inconsistent.

| # | File | Grain | Key columns |
|---|---|---|---|
| 1 | `players.csv` | 1 row per prospect (510) | `nfl_id`, `display_name`, `draft_year`, `nfl_position`, `birth_date`, `college_name`, `college_conference`, `draft_round`, `draft_pick_within_round`, `draft_overall_pick` (null = UDFA) |
| 2 | `combine_results.csv` | prospect × combine year | `combine_position`, `combine_height`, `combine_weight`, `hand_size`, `arm_length`, `wing_span`, `ten_yd_split`, `forty`, `vertical`, `broad_jump`, `three_cone`, `short_shuttle`, `bench_reps` (nulls = opt-outs), `ngs_athleticism_score`, `ngs_college_production_score`, `ngs_final_score` |
| 3 | `combine_tracking.csv` | 10 Hz frame, **6,310 drill attempts** | `draft_year`, `event_id` (one drill attempt), `nfl_id`, `entity_type` (`PLAYER`/`BALL`), `time` (ISO), `drill_type` (`FORTY_YARD_DASH`, `THREE_CONE_DRILL`, `SHORT_SHUTTLE`, `SKILL_DRILLS_WR/DB/OL/DL/TE`), `drill_name` (e.g., `GAUNTLET_DRILL`, `RUN_THE_HOOP_DRILL`, `BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION`, `SLOT_STRIKE_ROUTE_LEFT`, `PASS_PRO_MIRROR_DRILL`, `OL_PULL_DRILL_FOLD...`), `attempt`, `x`, `y`, `s`, `a`, `dis`, `dir`. **No `o` (orientation) column.** |
| 4 | `player_career_successes.csv` | 1 row per prospect | `career_offensive_snaps`, `career_defensive_snaps`, `career_special_teams_snaps`, `career_games_active`, `career_games_started`, `ap_all_pro_1st_team`, `ap_all_pro_2nd_team`, `pro_bowl_original_ballot` |
| 5 | `player_play.csv` | **314,197** player-plays (2023–2025, incl. PRE/POST) | Context: `game_id`, `play_id`, `team_abbr`, `nfl_id`, `lined_up_position`, down/distance/clock/score, `play_direction`. Value: pre-snap WP, WPA, `expected_points`, `expected_points_added`. Scheme: `offense_formation`, `receiver_alignment`, `pass_result`, `pass_length`, penalty fields, `yards_gained`, `team_coverage_man_zone`, `team_coverage_type`, `in_motion_at_ball_snap`. Receiving: `target`, `rec_yards`, `rush_yards`, `yards_after_catch`, `fumble(_lost)`, `route_ran`, **`separation_at_pass_forward`**, `cushion`, `expected_yards_after_catch`. OL: `dropback_duration`, `pass_rushers_encountered`, **`peak_pressure_probability_allowed`**, `pressure_allowed`, `sack_allowed`, `time_to_pressure_allowed`, `extended_sack_allowed`. DL/defense: `blitzing`, **`player_get_off`**, `sack`, `tackle`, `assist`, `tackle_for_loss`, `caused_forced_fumble`, `time_to_pressure`, `time_to_qb_hurry`, `coverage_assignment(_at_snap)`, `quick_pressure`, `unblocked_pressure` |
| 6 | `games.csv` | 1,002 games | `game_id`, `game_key`, `season`, `season_type` (`PRE`/`REG`/`POST`), `week`, `home_team_abbr`, `visitor_team_abbr` |
| 7–9 | `game_tracking_2023.csv`, `game_tracking_2024.csv`, `game_tracking_2025.csv` | 10 Hz frames | `game_id`, `play_id`, `nfl_id`, `time`, `x`, `y`, `s`, `a`, `dis`, `o`, `dir`, `event` (`ball_snap`, `pass_forward`, `pass_arrived`, `tackle`, `first_contact`, `out_of_bounds`, …) |

### Verified against the downloaded files (2026-10-15)
Full detail is in `docs/data_notes.md`, `docs/data_qa.md` and `docs/join_rates.md`. Corrections to the table above:
- **`player_play.csv` has 314,197 rows**, not 316,338. The player's team column is `team_abbr`.
- **Combine `event_id` is one player's attempt**, with a single prospect per attempt and never a drill partner. The `BALL` entity carries the receiver's `nfl_id`. `attempt` counts reps within a `drill_type` session, not within a drill. Most player × drill cells have one rep (75%); 24% have two.
- **Every prospect has his position's skill drills** (DB 122, OL 121, DL 118, WR 107, TE 42). Stopwatch drills are tracked only for a subset: 40 for 418 players, 3-cone for 160 and shuttle for 170. Four DL players have `SKILL_DRILLS_LB` attempts.
- Some drill names change spelling across years and are canonicalized in `drillCanon`. The OL mirror drill has 121 attempts (2023: `PASS_PRO_MIRROR_DRILL`, 2024–25: `PASS_PRO-MIRROR_DRILL`). DL `SHORT_ZONE_BREAKS` was split into three drills in 2025.

Structural notes [the earlier INFERENCES, now VERIFIED or corrected]:
1. **Game tracking covers only the cohort.** There are no non-cohort teammates, no opponents and no ball, and all `nfl_id`s are in `players.csv`. Each player-play starts 1.0 s before `ball_snap` and runs about 7 s, with `o` included. Up to 10 cohort players share a play. Separation, get-off and pressure must come from `player_play` NGS fields, plus the player's own movement.
2. **Unequal exposure** is confirmed. REG player-plays are 38.8k in 2023, 85.6k in 2024 and 124.9k in 2025, reflecting more classes in the league each year.
3. Draft-capital confounding: see `docs/eda/confounding.md`.
4. **n per group** is about 107–122 for WR/DB/OL/DL, and only 42 for TE.
5. Combine has no orientation and no frame id. Frames are exactly 10 Hz.
6. **PRE and POST are included.** `games.csv` has 49 PRE, 272 REG and 13 POST games per season, and PRE is 15–20% of player-plays. Always filter `season_type == "REG"`. nflverse has no preseason games, so PRE plays will not join to pbp.
7. The BALL entity in Combine appears only on WR/TE route and catch drills.
8. **NGS outcomes are position-specific:**
   - separation: WR/TE;
   - get-off: DL;
   - pressure probability allowed: OL;
   - cushion: DB and WR;
   - coverage_assignment: DB.
   There is no direct NGS coverage-success field for DBs.
9. **nflverse joins** match 100% for REG/POST games and plays (pbp, participation and FTN). 485 of 510 `nflId`s match nflverse `players`; the other 25 are fringe players with preseason-only snaps. Draft picks agree 100%. Every REG/POST cohort player-play is confirmed on the field in participation.

### Comparison with prior years [PRIOR YEAR, verified from those competitions' Kaggle pages]
- **2025:** `games.csv`, `plays.csv`, `players.csv`, `player_play.csv`, `tracking_week_[1-9].csv` (2022 season, all 22 players plus ball, camelCase columns, `frameId`, `frameType`). PFF columns had a `pff_` prefix.
- **2026:** `train/input_2023_w[01-18].csv` holds pre-throw tracking with `player_to_predict`, `player_role`, `ball_land_x/y`, and `num_frames_output`. `output_*` files hold ball-in-air x/y. There is also a supplementary play file.
- **New in 2027:** Combine tracking (first time ever), Combine measurements with NGS prospect scores, career accolades, multi-season (2023–25) cohort-only game tracking, and snake_case column names. Plays and player-play are merged into one rich `player_play.csv`.

---

## 3. Data-use rules [VERIFIED, Rules]
- **Data license: CC BY-NC 4.0.** Use is non-commercial only (competition, Kaggle forums, academic research and education). Do **not** redistribute the data to non-participants (Rules §2.4).
- **External data is allowed** (Rules §2.6) if it is "publicly available and equally accessible to use by all Participants … at no cost", or meets the "Reasonableness Standard" (minimal cost). The 2027 data page has no explicit nflverse note. 2025 and 2026 explicitly endorsed nflverse and Pro Football Reference [PRIOR YEAR]. nflverse (`nflreadpy`, already in requirements.txt) is therefore fine. Possible additions: draft/combine history from earlier years, snap counts, contracts, PFR stats. [UNVERIFIED] how `nfl_id` maps to nflverse `gsis_id`. Check for a crosswalk (nflverse `players` has `nfl_id`/`gsis_id` columns in some releases).
- **Code sharing:** No private sharing outside your team. Public sharing is allowed only on the competition's Kaggle forum or notebooks, and doing so licenses the code under an OSI license.
- **Winner license: Open Source** (an OSI-approved license that does not limit commercial use). Winners may need to provide reproducible code and methodology.
- **Open-source dependencies** must be OSI-licensed without commercial restrictions. AutoML tools are allowed if properly licensed.
- One Kaggle account per person.

---

## 4. What wins

### 4.1 Recent winners and finalists
**BDB 2023: linemen on pass plays** (tracks: Metric, Coaching, Undergrad). Source: allaccessfootball.com, U of T news.
- **Winner (Undergrad):** *Between the Lines: How Do We Measure Pressure?* by Inayatali, White, and Hocevar (U Toronto). A continuous pocket-pressure model with pocket-life-expectancy metrics (CPP, OPLE/DPLE, Surplus Pressure). The presentation featured **animated heat maps of the pocket**.
- Metric finalists:
  - *STRAIN* (Nguyen & Matthews): a simple, physics-inspired rate of the rusher closing distance to the QB. It is widely cited as a model notebook format.
  - *IDPI* (Bachelder)
  - *Player Impact Distribution* (Sekine & Sekine)
  - *Completions Added Through Suppression of Pressure* (Karpick)
- Coaching finalists: *Using Data to Determine Blitz Strategy* (Borsani) and *XPassRush: Identifying Pass Rushers Pre-Snap* (Ferraiola et al.).
- Undergrad finalist: *Open Space: Spatial Survival Probabilities* (UChicago).

**BDB 2024: tackling** (NFL.com finalist release).
- **Winner (Metric):** *Uncovering Missed Tackle Opportunities* by Chang, Dai, Jiang, and Cheng. A frame-level model of tackle probability in the next second. A *missed tackle opportunity* is when a defender's probability exceeds 75% for ≥0.5 s, then drops, and no tackle follows within 1 s. It found about 4× more missed opportunities than charted misses. Strengths: **an intuitive, coach-recognizable event definition**, plus clear rankings and visuals. In their post-mortem, the team said to keep the approach simple and budget serious time for the writeup.
- Other finalists:
  - *Momentum-Based Fractional Tackles* (Nguyen, Jiang, Ellingwood, Yurko; CMU): credits DL who slow the runner.
  - *No Edge, No Chance* (Hauck et al.; Coaching).
  - *Pull the Plug* (Davis & Rajendran; Coaching).
  - *Set a Framework to Evaluate Edge Setters* (Bajaj & Bhatia; NYU, Undergrad).

**BDB 2025: pre-snap to post-snap** (NFL finalist release, webwire mirror).
- **Winner (Undergrad):** *Exposing Coverage Tells in the Presnap* by Bajaj & Sandwar (NYU). A transformer (building on SumerSports' released model) predicts man/zone coverage frame by frame before the snap and shows how disguise "tells" emerge. They presented an **interactive "digital whiteboard"** where users can move players and see the prediction update.
- Other finalists:
  - *TEndencIQ* (Fleishman, Soriano, Steinberg, Ferraz; Undergrad): presented to defensive coaches.
  - *Safety Entropy* (Brill, Jacobson, Lipitz, Pipping; Metric).
  - *Under Cover 2: Predicting Disguised Defenses* (Pollack; Metric).
  - Ben Wendel (Coaching, slide deck).
- Semi-finalists: *Dialing Up the Pressure*, *Keep 'Em Separated*, *Down, Set, Hut*, *It's a Bluff: Predicting Receiver Decoy Motions*, *Trench Chess*.

**BDB 2026: ball in the air** (Analytics: University and Broadcast Visualization tracks; plus a separate RMSE prediction competition).
- **Winner:** *Ghostbusters: Back off man, I'm a Data Scientist!* by Lucca Ferraz (Rice; a 2025 finalist the year before). It compares each defender's movement while the ball is in the air to a distribution of hypothetical "ghost" defenders (conditional density estimation), producing **SLIME** scores. This is a process metric that credits good positioning regardless of outcome. Writeup: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/writeups/ghostbusters-back-off-man-im-a-data-scientist
- Other finalists [PARTIAL]: *Path Against the Pass* (Shah & Sullivan, UW-Madison, University track) and a BYU/UVU team (Nielson, Thompson, Miller, West). [UNVERIFIED] the full list and the Broadcast-track finalists. The list is at https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics/hackathon-winners, but it is JS-rendered and I could not retrieve it.

**Patterns across winners [INFERENCE]**
- One narrow, well-defined football question.
- A memorable name and acronym (STRAIN, SLIME, OPLE).
- Process over outcome: the metric credits *how* a player moves rather than box-score results.
- Strong animation or interactive visuals.
- A counterfactual or probability baseline (ghosts, tackle probability, pressure surfaces).
- Students win often. All three winners from 2023, 2025, and 2026 were students, so the University track is very competitive.

### 4.2 What the judges and organizers have said
- **NFL Analytics (Andrew Patton, Director of Analytical Research & Governance), via the AWS builder guide for BDB 2026:**
  - "Make sure you answer the prompt. Don't try and solve all of football."
  - Discuss limitations, since you have "a very narrow slice of data in a short time frame."
  - Graders read in their spare time, so formatting matters.
  - "Flowery prose is not your friend."
  - "Don't spend 200 hours on the technical part and 30 minutes on the write up."
  - Proofread.
  - "Pick a title that is memorable and informative."
  - Charts and legends should be readable without zooming.
  - "Your submission should look expensive."
  - Use correct football terminology.
  - Source: https://builder.aws.com/content/2oAsAXNgbmcR3gXCCTANimEePA7/learn-build-and-launch-with-aws-for-the-nfl-big-data-bowl-2026
- **Host prompt (every year):** focus narrowly. "In general, the more narrow the focus, the better the paper" (2025 evaluation page).
- **Ron Yurko (CMU, finalist/mentor), "Thoughts on Big Data Bowl 2025":**
  - "Always, always, always report base rates!"
  - Benchmark complex models against simple ones (for example, lasso).
  - "REPORT UNCERTAINTY!"
  - "you do not have as much data as you think"
  - "The prediction accuracy is just a minor point, while the novelty is in the downstream use of the model."
  - Source: https://statthinksportsanalytics.substack.com/p/thoughts-on-big-data-bowl-2025
- **SumerSports (data partner) on the 2025 win:** released models and data are a starting point, and "the real innovation was our own". Experiment with both modeling and presentation.
- [UNVERIFIED] I found no direct written guidance from Mike Lopez or Thompson (Tom) Bliss in this search. Bliss posts finalist announcements on X (@DataWithBliss), but the posts were not retrievable.

### 4.3 Winning-writeup checklist
**Gates (or you are not scored)**
- [ ] Explicitly links **Combine tracking**, not only stopwatch numbers, to **regular-season** NFL performance. Say this in the first paragraph.
- [ ] Uses the player tracking data.
- [ ] Attached public Kaggle Notebook, readable Markdown, embedded visuals, and a cover image.
- [ ] Submitted, not left as a draft, before **Jan 6, 2027, 23:59 UTC**. Track selected. All teammates on the team.

**Football (30%)**
- [ ] One position group and one or two drills. Answer one question a scout or position coach actually asks.
- [ ] Use the right football vocabulary (e.g., "sink hips", "get-off", "transition", "mirror", "stem").
- [ ] State how a team would use it during draft season *and* in season (player development, role fit, weekly matchup), since the rubric asks about "week-to-week" use.
- [ ] Account for football complexity: scheme (`team_coverage_type`, `route_ran`, `offense_formation`), alignment, role, snap share, and opponent.
- [ ] Include named-player examples. Judges know players like Jaxon Smith-Njigba and Will Anderson Jr.

**Data science (30%)**
- [ ] Control for draft capital and traditional Combine numbers. Show the **incremental** value of tracking features (ΔR², likelihood ratio, partial correlation).
- [ ] Use per-snap or per-opportunity outcomes. Handle unequal seasons (2025 class has 1 year). Use hierarchical or shrinkage models, and report base rates, CIs, and cross-validated uncertainty.
- [ ] Check reliability: attempt-to-attempt repeatability of the drill metric, and year-to-year stability of the game metric.
- [ ] Compare against a simple baseline. Avoid black boxes on n≈100.
- [ ] Validate face validity with known players.
- [ ] Include an honest limitations section (selection bias: only drafted or signed players; small n; drills differ by year).

**Writeup (20%)**
- [ ] A memorable title and acronym, plus a one-sentence thesis.
- [ ] Structure: hook → motivation → data → metric → validation → applications → limitations → appendix.
- [ ] ≤2,000 words (aim for about 1,700). Code hidden or in an appendix. Proofread by a football person.

**Visualization (20%)**
- [ ] Fewer than 10 figures and tables total, including GIFs. Count them.
- [ ] One hero animation: a Combine drill trace side by side with the same player's in-game rep.
- [ ] Colorblind-safe, large fonts, and direct labels instead of legends where possible. It should "look expensive".
- [ ] One scout-ready table (e.g., the top 10 prospects by metric with percentile bars).
- [ ] An innovative chart type tied to the metric (e.g., a deceleration-profile ribbon, a ghost overlay, or a draft-board heat strip).

---

## 5. Candidate project ideas

Scores are 1–5. "Novelty" is how unlike prior BDB and public work it is. "Feasibility" means doable in about 12 weeks with this data, given that game tracking is cohort-only.

### Idea 1: "Brake Check": deceleration-into-break efficiency (WR or DB)
- **Data:** `combine_tracking` WR route drills (`SLOT_STRIKE_ROUTE_*`, gauntlet) or DB `BACK_PEDAL_AND_TRANSITION_*`. Compute the peak deceleration before the cut, time spent below X yd/s, re-acceleration (0→peak a) after the `dir` change, and speed retained through the break. In-game, use `player_play.separation_at_pass_forward` by `route_ran`, plus the same break mechanics measured on game routes from `game_tracking`.
- **Metric:** a Break Efficiency Score (speed retained × re-acceleration ÷ decel distance) at the Combine. Its translation coefficient is measured against separation over expected (separation adjusted for route, cushion, coverage, and alignment).
- **Coach use:** Scouts get a sharper COD signal than the 3-cone. WR coaches get route-specific development targets, such as "loses 1.2 yd/s into breaks; drill sink-and-plant".
- **Novelty 4 · Feasibility 5.** This follows the host's first example closely, which lowers novelty but guarantees relevance.

### Idea 2: "Speed Tax": how much Combine speed shows up on Sundays
- **Data:** the 40-yard dash tracking (full speed and acceleration curve, not just time), and in-game max speed and acceleration by route or play type from `game_tracking`.
- **Metric:** the Game-Speed Retention %, defined as the share of a player's Combine top speed and acceleration profile he reaches in games, adjusted for role. Also, does the *shape* of the 40 curve (early burst vs. late top-end) predict game usefulness better than the final 40 time?
- **Coach use:** Flag "track-speed-only" prospects (bust risk) versus "plays faster than he tests". Coordinators can find vertical-threat or pursuit usage gaps.
- **Novelty 4 · Feasibility 5.** This works for every position group and gives a clean headline chart.

### Idea 3: "First Step to Pressure": DL burst transfer
- **Data:** the 10-yard split and the first 0.5–1.0 s of the 40 curve, DL skill drill acceleration onset, and `player_play.player_get_off`, `time_to_pressure`, `quick_pressure`, `sack` for DL/OLB on pass rushes.
- **Metric:** a Combine Burst Index (time to reach 3 and 5 yd/s; peak jerk). Model get-off and pressure rate per rush hierarchically, controlling for alignment and blitz.
- **Coach use:** Identify edge rushers whose burst will translate to quick pressure. Set a get-off development target for rookies.
- **Novelty 3 · Feasibility 5.** It pairs cleanly with the NGS get-off field.

### Idea 4: "Mirror Image": OL lateral mirror vs. pressure allowed
- **Data:** `PASS_PRO_MIRROR_DRILL` and OL agility drills (lateral speed, reaction lag, and overshoot when the direction flips), plus `peak_pressure_probability_allowed`, `pressure_allowed`, `time_to_pressure_allowed`, and `pass_rushers_encountered`.
- **Metric:** a Mirror Lag Score (ms between a direction change and matched velocity; overshoot in yards). The outcome is pressure probability allowed over expected, adjusted for dropback duration.
- **Coach use:** A guard vs. tackle projection, plus a vertical-set development focus. It flags tackles who should kick inside.
- **Novelty 5 · Feasibility 4.** OL is under-analyzed in BDB. One open question is whether the drill partner (the rusher) is tracked. Probably not, so measure the player only.

### Idea 5: "Trigger Time": DB reaction in drills vs. breaking on the ball in games
- **Data:** the DB `BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION` drill (time from pedal to transition, using heading change in `dir`, and drive speed). In-game DB frames after the `pass_forward` event: latency until peak acceleration and the drive toward the throw.
- **Metric:** Combine Trigger Time, linked to In-Game Ball-Break Latency, with outcomes such as completion allowed or PBU proxies when the DB is in coverage (`coverage_assignment`).
- **Coach use:** Assess zone vs. man fit (quick triggers in zone), and give DB coaches a reaction-time target.
- **Novelty 5 · Feasibility 4.** It needs careful event alignment. Game tracking has no ball, so use the event tags.

### Idea 6: "Movement Comps": nearest-neighbor movement fingerprints
- **Data:** all drills for one position group. Embed each attempt's speed, acceleration, and curvature time series (functional PCA or DTW), then find similar prospects and their game outcomes.
- **Metric:** a Movement Comp Score plus a comp-weighted projection of per-snap production, compared against an NGS-score-only baseline.
- **Coach use:** Front offices get data-driven player comps (a "moves like X" scouting language). Coaches get development plans copied from successful comps.
- **Novelty 4 · Feasibility 3.** It is visually striking, but validation is hard with small n.

### Idea 7: "Hidden in the Margins": tracking residual beyond the stopwatch
- **Data:** stopwatch results (`forty`, `three_cone`, `short_shuttle`, `ngs_athleticism_score`), tracking-derived features, `draft_overall_pick`, and per-snap outcomes.
- **Metric:** a Tracking Surplus, the part of tracking signal orthogonal to traditional tests and draft slot that predicts per-snap performance. Rank day-3 and UDFA players by surplus.
- **Coach use:** Direct value for finding hidden talent and avoiding busts, the abstract's exact wording.
- **Novelty 3 · Feasibility 4.** It is a strong framework section. Pair it with Idea 1, 4, or 5 rather than submitting it alone, since it risks being too broad.

### Idea 8: "Consistency Under Repetition": attempt-to-attempt variability
- **Data:** multiple `attempt`s per drill, and play-to-play variance in game metrics (get-off, separation).
- **Metric:** Drill Stability, the within-player coefficient of variation across attempts. Does it predict in-game consistency (low variance of separation or get-off) beyond mean ability?
- **Coach use:** Assess a floor-vs-ceiling risk profile, and give a "trust" signal for rookie playing time.
- **Novelty 5 · Feasibility 4.** It depends on how many attempts per player exist. Check this early.

### Recommendation [INFERENCE]
- **Top 3:** **#1 Brake Check** (WR/DB), **#4 Mirror Image** (OL), **#5 Trigger Time** (DB).
- Each answers one narrow scouting question, uses a named drill, has an NGS outcome in `player_play`, and supports a hero animation (drill trace beside a game rep).
- Use **#7's incremental-value framework** as the validation backbone of whichever you pick. **#2 Speed Tax** is the safest fallback.
- First data checks:
  - attempts per player per drill;
  - drill coverage per year;
  - whether game tracking truly excludes non-cohort players;
  - the `REG` filter;
  - n per position.
