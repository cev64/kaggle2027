# Project recommendation (after the real-data deep dive)

*2026-10-15. Evidence: `docs/eda/drill_quality.md`, `docs/eda/outcomes.md`, `docs/eda/game_movement.md`, `docs/eda/confounding.md`, and `scripts/eda/incremental_checks.py`.*

## What the data does and does not support

- **Combine tracking is accurate.**
  - Tracked 40 times match the official ones (r = 0.98 when measured from 2 yd).
  - The 10–40 split matches the official one (r = 0.98), 3-cone matches at r = 0.90 and shuttle at r = 0.83.
  - The reported `s` matches speed derived from positions (r = 0.999).
  - Caveats:
    - `a` is unsigned and `dis` is not the per-frame step.
    - Clips are trimmed inconsistently: 48% start with the player already moving, and clip lengths change by year.
- **Most skill drills give one rep, and their change-of-direction features are not player traits.**
  - Brake and COD features do not agree across routes or DB drills (alpha −0.1 to 0.3).
  - DB transition turn time has a retest reliability of 0.13.
  - Rep-to-rep variability has a split-half of about 0.
- **The reliable tracking-only signals**, which are also nearly orthogonal to the stopwatch:
  - **40-curve shape**: 2–10 yd split (reliability 0.77), distance to 90% of top speed (0.71) and v10/vmax (0.64). Forty plus weight explain at most 30% of each.
  - **OL mirror reversal** "dead time" (0.84) and peak lateral braking (0.76). These have |r| ≤ 0.25 with the stopwatch and with NGS athleticism.
- **Drill-to-game bridge.**
  - Only **speed** clearly translates. Combine 40 peak speed vs in-game p99 speed: WR r = 0.65 [0.47, 0.77], DL 0.72, TE 0.57, DB 0.32, OL 0.31.
  - Rookies reach 74–90% of their tested top speed, depending on group.
  - Route-break, DB-trigger and OL-set bridges are all |r| ≤ 0.26.
- **Outcomes.**
  - **DL get-off is the only NGS outcome that is both reliable and skill-like.** It needs 14 rush snaps for reliability 0.7, and its year-to-year r is 0.86. It is largely a re-measure of in-game displacement at +1 s (r = 0.90).
  - WR separation over expected and OL pressure probability over expected are mostly context: year-to-year r falls from about 0.7–0.8 raw to about 0.4 after adjustment, and they need 300–430 plays for reliability 0.7.
  - DBs have no usable coverage outcome.
- **Confounding and selection.**
  - Draft slot explains 37–64% of log snaps. Only 55–60% of each group reaches a usable per-snap sample, and few of those are UDFAs.
  - Stopwatch and size predict draft slot at cross-validated R² of about 0. `ngs_final_score` is effectively a draft-grade proxy (ρ ≈ −0.75 with log pick).
- **Player-level incremental value is not detectable at this n.**
  - Across 267 exploratory tests, none reaches q < 0.10.
  - My targeted checks (DL n = 77, WR n = 73) also show nothing: 40-curve shape and in-game speed vs get-off, pressure over expected, separation over expected and targets/route all have |partial r| ≤ 0.20 after controlling for size, role, log pick and forty.
  - The only hit is that distance to 90% of top speed predicts DL time-to-pressure (on pressures only): partial r = −0.42, p < 0.001, n = 70. That outcome rests on about 11 pressures per player, so treat it as a lead, not a finding.

**Implication.** A writeup claiming "Combine drill X predicts rookie outcome Y beyond the stopwatch and draft slot" at the player level (n ≈ 70) would be underpowered and probably null. The defensible and more original approach is to model *translation at the play level*. That means using each player's Combine curve as his personal **capacity ceiling**, then measuring, play by play, how much of that capacity he uses on Sundays and what the usage buys. That design has thousands of plays per group, uses within-player variation (which removes draft slot and talent confounding), and still links Combine tracking to regular-season performance.

## Ranking of the 8 candidate ideas

| rank | idea | drill side | game bridge | outcome side | verdict |
|---|---|---|---|---|---|
| 1 | **Speed Tax** (Combine speed/accel capacity → game usage) | strong (40 reliable; curve shape orthogonal to the time) | **strong** (r 0.65–0.72 WR/DL) | ok (in-game speed reliable; separation is noisy per player, but play-level within-player variation is usable) | **Flagship** |
| 2 | **First Step** (DL burst → get-off/pressure) | ok (40 curve; DL skill drills unreliable) | ok (10-yd split ↔ get-off r ≈ 0.35–0.39) | **strong** (get-off) | **Backup**, in the same "capacity" framework |
| 3 | Hidden in the Margins (tracking surplus) | — | — | — | Use as the validation and robustness section, not as a project |
| 4 | Mirror Image (OL) | ok/novel (reliable, orthogonal) | weak (no link to game set or pressure) | weak (pressure over expected needs about 300 snaps) | Possible one-figure sidebar; high risk of a null |
| 5 | Brake Check (WR/DB) | weak (single reps, cuts inconsistent across routes) | weak (≤ 0.16) | ok for WR / none for DB | Drop |
| 6 | Movement Comps | weak (single reps) | — | weak | Drop (cannot be validated at n ≈ 100) |
| 7 | Trigger Time (DB) | weak (reliability 0.13) | weak | none | Drop |
| 8 | Consistency | not a trait (split-half ≈ 0) | — | weak | Drop |

## Flagship: "Speed Tax": how much of his Combine speed a WR actually plays with, and what it buys

**Thesis.** The tracked 40 measures each receiver's personal speed and acceleration ceiling. In the NFL, rookies run routes at only part of that ceiling, and the share they use (not the ceiling itself) is what earns separation.

**Scope.** WRs: 107 prospects, 94 with a tracked 40, 63–83 with enough regular-season routes. TEs are pooled in as a second level. DL appears as one comparison panel, which reuses the backup's pipeline.

**Modeling plan.**
1. *Capacity profile* from the 40, for each player:
   - top speed;
   - an acceleration curve fit by a mono-exponential, v(t) = vmax(1 − e^{−t/τ}), giving τ;
   - distance to 90% of top speed;
   - 2–10 yd split.
   Fix the trimming issue by timing from 2 yd. Report retest reliability (≥ 0.7 for all four), and show these add information beyond the stopwatch forty (R² with forty + weight ≤ 0.3).
2. *Capacity used per play.* On each route (REG, from the snap to pass_forward), measure peak speed and acceleration onset as a fraction of the player's own Combine curve. A hierarchical model (route type × alignment × coverage × down/distance, with player random effects) separates a player's **tax** (his habitual shortfall) from situational demand.
3. *What it buys.* Within player, model play-level `separation_at_pass_forward` (and target probability) on capacity used, using player fixed effects and the context covariates. Draft slot and talent cancel out within a player, which answers the confounding critique directly.
4. *Beyond the stopwatch and draft slot*, at the player level, honestly reported:
   - nested cross-validation comparing (draft slot + forty + size) with (+ tracked curve features) for predicting in-game speed and the tax;
   - ΔR² and bootstrap CIs, with power stated, and the null shown if it is null;
   - IPW for selection into the route sample as a sensitivity check.
5. *Uncertainty and shrinkage.* Use empirical-Bayes or hierarchical estimates for each player's tax, and report base rates and credible intervals.

**Football use.**
- *Draft season:* the curve shape (burst vs long speed) beyond the 40 time, plus how much of that speed comparable profiles have historically used.
- *In season, week to week:* a WR's tax as a monitoring signal for route-running effort, injury return or fatigue, and a development target ("play at 88% of capacity on vertical stems").

**Figure budget (≤ 9).**
1. A hero animation: the player's 40 curve overlaid on his fastest game rep.
2. Validation of tracked vs official 40.
3. A capacity curve ribbon by group.
4. A "tax" distribution by route.
5. Within-player separation vs capacity used.
6. Incremental-value forest plot.
7. A scout table (top and bottom 10).
8. A DL comparison panel.

Aim for 1,700 words.

## Backup: "First Step" (DL), within the same capacity framework

The design is the same, applied to DL. The capacity is the 40's first 10 yd (2–10 split, distance to 90% of top speed). The play-level usage is the displacement or speed at +0.5 s / +1 s after the snap (which re-measures NGS get-off, r = 0.90). The payoffs are pressure and time-to-pressure, with role (interior vs edge), weight and alignment controlled.

Strengths:
- It has the most reliable outcome in the data (get-off).
- It has the best non-selected sample (76 DL with ≥ 50 rush snaps).
- It has one promising lead: distance to 90% of top speed vs time-to-pressure, partial r = −0.42.

Risk: get-off is heavily explained by size and role, so the Combine tracking surplus may be null.

## Timeline to Jan 6, 2027 (about 12 weeks)

| weeks | work |
|---|---|
| 1–2 (to Oct 29) | Capacity curves and fixes (2-yd timing, curve fits), the play-level route feature table |
| 3–5 (to Nov 19) | Hierarchical and within-player models, incremental-value CV, DL panel |
| 6–7 (to Dec 3) | Robustness (IPW, class-year, route subsets), named-player face validity |
| 8–10 (to Dec 24) | Figures and animation, then the Kaggle notebook |
| 11–12 (to Jan 6) | Writeup (≤ 2,000 words, < 10 figures), proofreading by a football person, submit by Jan 4 for buffer |
