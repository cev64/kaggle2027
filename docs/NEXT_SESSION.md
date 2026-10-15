# Next-session handoff prompt

Copy everything below the line into a new Claude Code session on this repo/environment.

---

Continue the NFL Big Data Bowl 2027 project on branch `claude/amazing-curie-he8ne9` (repo cev64/kaggle2027). Check out that branch and read `docs/competition_brief.md`, `docs/external_data.md`, and any other files in `docs/` first; they hold the competition rules, judging criteria, past-winner analysis, candidate project ideas and nflverse join-key notes from the previous session. The code lives in `src/bdb/` (external.py, keys.py, tracking.py, clean.py, features.py, viz.py), with tests in `tests/` and scripts in `scripts/`.

Steps:
1. `pip install -r requirements.txt`, then confirm `KAGGLE_API_TOKEN` is set (don't print it) and run `./scripts/download_kaggle.sh`. If it returns 401/403, tell me to accept the competition rules on Kaggle or fix the token, and stop.
2. Inspect every file in `data/raw/`: schemas, row counts, seasons and weeks, frame rates, new columns compared with the schemas assumed in `src/bdb/tracking.py` and `docs/competition_brief.md`. Update the brief's data section with the real schema.
3. Adapt `src/bdb/tracking.py` and `src/bdb/clean.py` to the real schema, run `scripts/build_processed.py` to produce cleaned parquet files plus `docs/data_qa.md`, and keep `pytest` green.
4. Run `scripts/fetch_external.py` if `data/external/` is empty, then use `src/bdb/keys.py` to verify the join rates from the Big Data Bowl data to nflverse (gameId, playId, nflId) and report them.
5. Using subagents in parallel, run an exploratory deep dive aligned with the 2027 theme. Then rank the candidate ideas in the brief against what the real data supports, and recommend one flagship project and one backup, with a short rationale and a modeling plan.
6. Commit and push to the same branch as you go. Report back with the data summary, QA findings, join rates and the recommendation before starting heavy modeling or the front-end writeup.
