# Next-session handoff prompt

Copy everything below the line into a new Claude Code session on this repo/environment.

---

Continue the NFL Big Data Bowl 2027 project on branch `claude/amazing-curie-he8ne9` (repo cev64/kaggle2027). Check out that branch and read `docs/competition_brief.md`, `docs/external_data.md`, and any other files in `docs/` first; they hold the competition rules, judging criteria, past-winner analysis, candidate project ideas and nflverse join-key notes from the previous session. The code lives in `src/bdb27/` (external.py, keys.py, tracking.py, clean.py, drills.py, features.py, viz.py), with tests in `tests/` and scripts in `scripts/`.

Steps:
1. `pip install -r requirements.txt`, then confirm `KAGGLE_API_TOKEN` is set (don't print it) and run `./scripts/download_kaggle.sh`. If it returns 401/403, tell me to accept the competition rules on Kaggle or fix the token, and stop.
2. The 2027 theme is "Combine to NFL Performance": link 10 Hz Combine drill tracking for 510 rookies (2023-25 classes) to regular-season results. Inspect every file in `data/raw/` (combine_tracking, combine_results, players, player_career_successes, player_play, games, game_tracking_2023-2025): schemas, row counts, drills and attempts per player, frame rates, which entities appear in game tracking, and whether games include preseason. Confirm or correct the data section of `docs/competition_brief.md`.
3. Adapt `src/bdb27/tracking.py` and `src/bdb27/clean.py` to the real schema, run `scripts/build_processed.py` to produce cleaned parquet files plus `docs/data_qa.md`, and keep `pytest` green.
4. Run `scripts/fetch_external.py` if `data/external/` is empty, then use `src/bdb27/keys.py` to verify the join rates from the Big Data Bowl data to nflverse (gameId, playId, nflId) and report them.
5. Using subagents in parallel, run an exploratory deep dive: drill coverage by position group (WR, TE, DB, OL, DL), the quality of drill kinematics, the outcome metrics available in player_play (separation, get-off, pressure probability allowed) and their reliability by sample size, and confounding by draft slot. Then rank the 8 candidate ideas in the brief (e.g. Brake Check, Mirror Image, Trigger Time, Speed Tax) against what the real data supports, and recommend one flagship project and one backup, with a modeling plan centred on value beyond stopwatch times and draft slot. Remember the 2,000-word limit, fewer than 10 figures or tables, and the January 6, 2027 deadline.
6. Commit and push to the same branch as you go. Report back with the data summary, QA findings, join rates and the recommendation before starting heavy modeling or the front-end writeup.
