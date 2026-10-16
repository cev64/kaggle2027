# What explains the shrinkage? Moderators of Combine-to-game speed transfer

_Generated 2026-10-16 by `scripts/shrinkage.py`. Raw estimates: `data/processed/eda/shrinkage_estimates.csv`, tests: `data/processed/eda/shrinkage_tests.csv`._

## Summary: what explains the shrinkage

- **Not who the player is, or when.** Across 80 `logCapC × moderator` tests (role, alignment, draft
  capital, experience, usage, speed-curve shape, quarter, score margin, route depth), 7 have p < 0.05 and
  **none survives Benjamini-Hochberg** (smallest q = 0.17). The WR slot > outside hit and the DB role
  pattern flip sign between the window and whole-play designs.
- **Experience does not raise transfer.** Within player (same player, rookie vs year 2-3, player FE) the change
  in elasticity is WR +0.12 [-0.15, 0.38], DB +0.10 [-0.27, 0.47], DL -0.12 [-0.26, 0.02] in windows, and about 0
  in whole-play p99.9. High picks do not keep more of their speed. Burst shape (v10rel) does not moderate. The
  fast-slow gap does not shrink in Q4 or in blowouts (Q4 point estimates are, if anything, slightly positive).
- **Combine measurement error explains little: 1-7% of the shortfall.** Reliability would need to be about 0.27-0.42
  to explain all of it. The observed reliability is 0.88-0.98.
- **Headroom: game speed is demand-limited.** Players never run at Combine speed. Their fastest 0.1% of REG
  frames reach a median 82-85% of vmax (WR/DB) and 73-75% (DL). Essentially no frames reach 90%. For DB and DL,
  Combine vmax barely orders *typical* game speed (p90 elasticity DB 0.12, DL -0.17). It shows up only in the
  rare all-out frames (p99.9: DB 0.48, DL 0.51; p99.99: DB 0.34, DL 0.65), and even there it stays well below 1.
  The capacity is there, but few plays call for all of it, and the play (route, angle, pursuit target) sets the
  speed. (WR is the exception: transfer is higher at p90, 0.53, probably because faster WRs are assigned deeper
  routes. That is role, not capacity.)
- **Exploratory, same direction in both designs (not BH-significant): DL.** Edge rushers keep more speed than
  interior linemen (windows 0.73 vs 0.06; whole-play p99.9 0.77 vs 0.41). High-usage DL keep more than rotational
  DL (windows +0.50 per SD of snaps/game, p = 0.01; whole-play +0.23, p = 0.08). Edge role and snap share overlap,
  so these may be one effect. Interior linemen meet a blocker within a yard or two, so top speed rarely binds.

**Writeup-worthy.** (1) *Speed on Sundays is set by the play, not the engine.* Even on their fastest frames,
players use only about 80% of their Combine top speed. Capacity separates players only in those rare all-out
moments. That headroom, not stopwatch noise (< 7% of the gap), is why transfer is about 40 cents on the dollar.
(2) *A stable discount:* transfer does not grow with experience (tested within player), draft slot or Q4/blowout
game state. The 0.3-0.5 elasticity looks like a fixed property of each position, not a rookie adjustment or a
fatigue effect.

## Design

Baseline: within group, log(window peak speed) ~ log(Combine vmax) + Combine weight + the `translation.py` context + window-length spline + (1 | player); tracked 40s only. Baseline elasticities here: WR 0.27 [0.06, 0.47], DB 0.39 [0.04, 0.75], DL 0.42 [0.07, 0.77].

- **Player-level moderators** (role, draft tier, usage, v10rel): `logCapC × moderator` and `weight × moderator` in the MixedLM, so each stratum's slope is weight-controlled within that stratum. Replicated on per-player whole-play p99 / p99.9 of all REG frames (OLS, HC1), players with >= 200 plays.
- **Play/season-level moderators** (experience, quarter, score margin, alignment, route depth): OLS with player fixed effects and player-clustered SEs. The vmax main effect is absorbed, so the `logCapC × moderator` term is identified purely *within player* (same player, rookie vs later season; Q1-3 vs Q4). Whole-play versions use player × cell p99 / p99.9 with player FE.
- Role: WR outside vs SLOT_WR, and DL EDGE vs INTERIOR_LINE, by majority `linedUpPosition`; DB by `nfl_position` (FS/SS = safety) with CBs whose majority alignment is SLOT_CB as slot CB. Draft tier from `draft_round` (null = UDFA). Usage = snaps per REG game (routes per game for WR) from `player_play`. Experience = season - draft year + 1 (only the 2023 and 2024 classes have a later season). Margin = own-team pre-snap score margin.
- **Multiplicity.** 80 interaction tests (joint Wald for multi-level moderators), Benjamini-Hochberg across all of them.

## Interaction estimates (Δ elasticity vs reference level)

`windows` = snap-to-throw peak speed; `whole-play p99.9` = per-player fastest 0.1% of REG frames. p and q are for the joint test of the moderator within the group.

| moderator                                     | group   | contrast                 | windows: Δ elasticity [95% CI]   |   windows p |   windows q (BH) | whole-play p99.9: Δ [95% CI]   |   whole-play q (BH) | same sign   |
|:----------------------------------------------|:--------|:-------------------------|:---------------------------------|------------:|-----------------:|:-------------------------------|--------------------:|:------------|
| a. role (player majority)                     | WR      | slot - outside           | +0.48 [+0.06, +0.91]             |       0.026 |            0.413 | -0.09 [-0.76, +0.57]           |            0.9728   | no          |
| a. role (player majority)                     | DB      | slot CB - outside CB     | -1.30 [-3.12, +0.51]             |       0.22  |            0.746 | +0.70 [-0.12, +1.52]           |            0.284054 | no          |
| a. role (player majority)                     | DB      | safety - outside CB      | +0.30 [-0.46, +1.06]             |       0.22  |            0.746 | -0.59 [-1.28, +0.10]           |            0.284054 | no          |
| a. role (player majority)                     | DL      | interior - edge          | -0.67 [-1.28, -0.06]             |       0.031 |            0.413 | -0.36 [-1.00, +0.28]           |            0.813946 | yes         |
| a. alignment on the play                      | WR      | slot - outside           | +0.10 [-0.18, +0.39]             |       0.475 |            0.939 |                                |                     |             |
| a. alignment on the play                      | DB      | slot CB - outside CB     | -0.19 [-1.64, +1.26]             |       0.562 |            0.956 |                                |                     |             |
| a. alignment on the play                      | DB      | high safety - outside CB | +0.27 [-1.33, +1.87]             |       0.562 |            0.956 |                                |                     |             |
| a. alignment on the play                      | DL      | interior - edge          | -0.28 [-0.63, +0.07]             |       0.121 |            0.644 |                                |                     |             |
| b. draft capital                              | WR      | R3-4 - R1-2              | +0.21 [-0.46, +0.87]             |       0.684 |            0.959 | +0.16 [-0.51, +0.83]           |            0.9728   | yes         |
| b. draft capital                              | WR      | R5-7/UDFA - R1-2         | -0.03 [-0.58, +0.51]             |       0.684 |            0.959 | +0.10 [-0.61, +0.82]           |            0.9728   | no          |
| b. draft capital                              | DB      | R3-4 - R1-2              | +0.37 [-0.52, +1.26]             |       0.715 |            0.973 | -0.24 [-0.86, +0.38]           |            0.956946 | no          |
| b. draft capital                              | DB      | R5-7/UDFA - R1-2         | +0.15 [-0.76, +1.07]             |       0.715 |            0.973 | -0.34 [-1.38, +0.71]           |            0.956946 | no          |
| b. draft capital                              | DL      | R3-4 - R1-2              | -0.61 [-1.85, +0.64]             |       0.305 |            0.856 | -0.54 [-1.67, +0.58]           |            0.956201 | yes         |
| b. draft capital                              | DL      | R5-7/UDFA - R1-2         | -1.00 [-2.32, +0.31]             |       0.305 |            0.856 | -0.39 [-1.63, +0.85]           |            0.956201 | yes         |
| c. experience, within player                  | WR      | year 2-3 - rookie        | +0.12 [-0.15, +0.38]             |       0.383 |            0.937 | -0.02 [-0.73, +0.69]           |            0.994471 | no          |
| c. experience, within player                  | DB      | year 2-3 - rookie        | +0.10 [-0.27, +0.47]             |       0.6   |            0.956 | +0.11 [-0.59, +0.81]           |            0.9728   | yes         |
| c. experience, within player                  | DL      | year 2-3 - rookie        | -0.12 [-0.26, +0.02]             |       0.083 |            0.524 | +0.04 [-0.36, +0.44]           |            0.9728   | no          |
| c. experience, between+within                 | WR      | year 2-3 - rookie        | +0.11 [-0.05, +0.28]             |       0.181 |            0.726 |                                |                     |             |
| c. experience, between+within                 | DB      | year 2-3 - rookie        | +0.11 [-0.19, +0.42]             |       0.471 |            0.939 |                                |                     |             |
| c. experience, between+within                 | DL      | year 2-3 - rookie        | -0.00 [-0.19, +0.18]             |       0.973 |            0.994 |                                |                     |             |
| d. usage tier (median split)                  | WR      | high usage - low usage   | -0.13 [-0.54, +0.28]             |       0.521 |            0.939 | -0.39 [-0.95, +0.18]           |            0.725999 | yes         |
| d. usage tier (median split)                  | DB      | high usage - low usage   | +0.06 [-0.64, +0.77]             |       0.858 |            0.973 | -0.58 [-1.24, +0.07]           |            0.523952 | no          |
| d. usage tier (median split)                  | DL      | high usage - low usage   | +1.03 [+0.37, +1.69]             |       0.002 |            0.17  | +0.11 [-0.50, +0.73]           |            0.9728   | yes         |
| d. usage, continuous (per SD)                 | WR      | per +1 SD                | -0.03 [-0.25, +0.20]             |       0.828 |            0.973 | -0.15 [-0.49, +0.19]           |            0.936706 | yes         |
| d. usage, continuous (per SD)                 | DB      | per +1 SD                | -0.19 [-0.57, +0.18]             |       0.31  |            0.856 | -0.30 [-0.60, -0.00]           |            0.523952 | yes         |
| d. usage, continuous (per SD)                 | DL      | per +1 SD                | +0.50 [+0.12, +0.87]             |       0.01  |            0.284 | +0.23 [-0.03, +0.49]           |            0.523952 | yes         |
| e. relative 10-yd speed v10rel (per SD)       | WR      | per +1 SD                | +0.15 [-0.02, +0.32]             |       0.081 |            0.524 | -0.03 [-0.27, +0.22]           |            0.9728   | no          |
| e. relative 10-yd speed v10rel (per SD)       | DB      | per +1 SD                | +0.12 [-0.25, +0.49]             |       0.528 |            0.939 | -0.03 [-0.40, +0.33]           |            0.9728   | no          |
| e. relative 10-yd speed v10rel (per SD)       | DL      | per +1 SD                | -0.09 [-0.35, +0.16]             |       0.476 |            0.939 | +0.01 [-0.17, +0.19]           |            0.994471 | no          |
| f. Q4/OT vs Q1-3, within player               | WR      | Q4/OT - Q1-3             | -0.00 [-0.15, +0.15]             |       0.994 |            0.994 | +0.24 [-0.08, +0.57]           |            0.725999 | no          |
| f. Q4/OT vs Q1-3, within player               | DB      | Q4/OT - Q1-3             | +0.23 [-0.10, +0.56]             |       0.167 |            0.726 | +0.38 [-0.15, +0.90]           |            0.725999 | yes         |
| f. Q4/OT vs Q1-3, within player               | DL      | Q4/OT - Q1-3             | +0.04 [-0.09, +0.17]             |       0.547 |            0.951 | -0.08 [-0.41, +0.25]           |            0.956201 | no          |
| f. margin >= 14 vs closer, within player      | WR      | margin >= 14 - within 13 | -0.05 [-0.24, +0.15]             |       0.645 |            0.956 | +0.01 [-0.43, +0.45]           |            0.994471 | no          |
| f. margin >= 14 vs closer, within player      | DB      | margin >= 14 - within 13 | +0.44 [-0.02, +0.90]             |       0.063 |            0.524 | +0.08 [-0.64, +0.80]           |            0.9728   | yes         |
| f. margin >= 14 vs closer, within player      | DL      | margin >= 14 - within 13 | -0.03 [-0.17, +0.11]             |       0.67  |            0.957 | +0.11 [-0.32, +0.53]           |            0.956201 | no          |
| f. route depth / deep coverage, within player | WR      | intermediate - short     | +0.03 [-0.31, +0.38]             |       0.217 |            0.746 |                                |                     |             |
| f. route depth / deep coverage, within player | WR      | vertical - short         | +0.16 [-0.16, +0.48]             |       0.217 |            0.746 |                                |                     |             |
| f. route depth / deep coverage, within player | DB      | deep - underneath        | +0.13 [-0.23, +0.48]             |       0.49  |            0.939 |                                |                     |             |

## Stratum-specific elasticities

| group   | moderator   | level      | windows             | whole-play p99.9    |   players (windows) |
|:--------|:------------|:-----------|:--------------------|:--------------------|--------------------:|
| DB      | role        | outside CB | 0.28 [-0.21, 0.77]  | 0.55 [0.09, 1.01]   |                  27 |
| DB      | role        | safety     | 0.58 [-0.00, 1.16]  | -0.04 [-0.56, 0.47] |                  27 |
| DB      | role        | slot CB    | -1.02 [-2.77, 0.73] | 1.25 [0.57, 1.93]   |                   7 |
| DB      | draftTier   | R1-2       | 0.17 [-0.47, 0.80]  | 0.67 [0.27, 1.06]   |                  15 |
| DB      | draftTier   | R3-4       | 0.54 [-0.10, 1.18]  | 0.42 [-0.05, 0.90]  |                  21 |
| DB      | draftTier   | R5-7/UDFA  | 0.32 [-0.33, 0.98]  | 0.33 [-0.64, 1.30]  |                  27 |
| DB      | expTier     | rookie     | 0.25 [-0.14, 0.64]  |                     |                  62 |
| DB      | expTier     | year 2-3   | 0.36 [-0.05, 0.78]  |                     |                  46 |
| DB      | usageTier   | high usage | 0.48 [0.00, 0.96]   | 0.18 [-0.19, 0.56]  |                  32 |
| DB      | usageTier   | low usage  | 0.42 [-0.12, 0.95]  | 0.77 [0.23, 1.30]   |                  31 |
| DL      | role        | edge       | 0.73 [0.28, 1.18]   | 0.77 [0.25, 1.29]   |                  28 |
| DL      | role        | interior   | 0.06 [-0.35, 0.47]  | 0.41 [0.03, 0.78]   |                  33 |
| DL      | draftTier   | R1-2       | 1.00 [-0.14, 2.15]  | 0.94 [-0.10, 1.98]  |                  15 |
| DL      | draftTier   | R3-4       | 0.40 [-0.10, 0.89]  | 0.40 [-0.04, 0.83]  |                  20 |
| DL      | draftTier   | R5-7/UDFA  | -0.00 [-0.66, 0.66] | 0.55 [-0.13, 1.23]  |                  26 |
| DL      | expTier     | rookie     | 0.39 [0.02, 0.77]   |                     |                  60 |
| DL      | expTier     | year 2-3   | 0.39 [0.02, 0.76]   |                     |                  44 |
| DL      | usageTier   | high usage | 0.89 [0.41, 1.36]   | 0.61 [0.14, 1.07]   |                  31 |
| DL      | usageTier   | low usage  | -0.14 [-0.59, 0.31] | 0.49 [0.09, 0.90]   |                  30 |
| WR      | role        | outside    | 0.09 [-0.12, 0.31]  | 0.29 [0.05, 0.53]   |                  48 |
| WR      | role        | slot       | 0.58 [0.21, 0.95]   | 0.19 [-0.43, 0.82]  |                  15 |
| WR      | draftTier   | R1-2       | 0.19 [-0.27, 0.66]  | 0.17 [-0.44, 0.78]  |                  15 |
| WR      | draftTier   | R3-4       | 0.40 [-0.07, 0.87]  | 0.33 [0.06, 0.60]   |                  22 |
| WR      | draftTier   | R5-7/UDFA  | 0.16 [-0.12, 0.44]  | 0.27 [-0.10, 0.64]  |                  27 |
| WR      | expTier     | rookie     | 0.24 [0.02, 0.46]   |                     |                  64 |
| WR      | expTier     | year 2-3   | 0.35 [0.12, 0.58]   |                     |                  41 |
| WR      | usageTier   | high usage | 0.17 [-0.14, 0.48]  | 0.06 [-0.40, 0.52]  |                  32 |
| WR      | usageTier   | low usage  | 0.30 [0.03, 0.57]   | 0.45 [0.12, 0.77]   |                  32 |

## Measurement error in the Combine vmax

Classical error in the regressor attenuates the slope by the reliability of the player's Combine mean (within-group retest ICC, Spearman-Brown for the mean number of reps).

| group   |   elasticity (weight-ctrl) |   vmax reliability |   disattenuated |   share of shortfall (1 - b) due to Combine error |   reliability needed to explain all |
|:--------|---------------------------:|-------------------:|----------------:|--------------------------------------------------:|------------------------------------:|
| WR      |                      0.265 |              0.877 |           0.303 |                                             0.051 |                               0.265 |
| DB      |                      0.394 |              0.904 |           0.436 |                                             0.069 |                               0.394 |
| DL      |                      0.423 |              0.984 |           0.43  |                                             0.012 |                               0.423 |

## Ceiling / headroom: how close do players get to their Combine top speed?

Per-player frame-speed percentiles over all REG frames (incl. after the catch and in pursuit), pooled across seasons, regressed on log vmax + log weight (HC1).

| group   | quantile   |   elasticity |   lo95 |   hi95 |   game/Combine median |   players |
|:--------|:-----------|-------------:|-------:|-------:|----------------------:|----------:|
| WR      | p50        |        0.984 |  0.388 |  1.58  |                 0.317 |        48 |
| WR      | p90        |        0.525 |  0.252 |  0.798 |                 0.651 |        48 |
| WR      | p99        |        0.438 |  0.238 |  0.638 |                 0.782 |        48 |
| WR      | p999       |        0.294 |  0.021 |  0.567 |                 0.839 |        48 |
| WR      | p9999      |        0.186 | -0.401 |  0.773 |                 0.863 |        48 |
| DB      | p50        |       -0.279 | -1.028 |  0.47  |                 0.23  |        55 |
| DB      | p90        |        0.118 | -0.16  |  0.396 |                 0.526 |        55 |
| DB      | p99        |        0.366 |  0.091 |  0.641 |                 0.724 |        55 |
| DB      | p999       |        0.478 |  0.132 |  0.823 |                 0.826 |        55 |
| DB      | p9999      |        0.337 |  0.024 |  0.651 |                 0.865 |        55 |
| DL      | p50        |       -0.015 | -0.564 |  0.534 |                 0.2   |        51 |
| DL      | p90        |       -0.166 | -0.592 |  0.259 |                 0.449 |        51 |
| DL      | p99        |        0.069 | -0.219 |  0.356 |                 0.631 |        51 |
| DL      | p999       |        0.51  |  0.19  |  0.83  |                 0.742 |        51 |
| DL      | p9999      |        0.651 |  0.251 |  1.051 |                 0.79  |        51 |



| group   | Combine vmax tercile   |   players |   vmax (yd/s) |   p99.9 game (yd/s) |   p99.9 / vmax |   % frames >= 80% vmax |
|:--------|:-----------------------|----------:|--------------:|--------------------:|---------------:|-----------------------:|
| WR      | slowest third          |        16 |        11.099 |               9.372 |          0.852 |                  0.768 |
| WR      | middle                 |        16 |        11.47  |               9.539 |          0.836 |                  0.433 |
| WR      | fastest third          |        16 |        11.935 |               9.87  |          0.821 |                  0.336 |
| DB      | slowest third          |        19 |        11.156 |               9.405 |          0.841 |                  0.372 |
| DB      | middle                 |        18 |        11.466 |               9.37  |          0.82  |                  0.197 |
| DB      | fastest third          |        18 |        11.837 |               9.639 |          0.814 |                  0.128 |
| DL      | slowest third          |        17 |         9.581 |               7.146 |          0.754 |                  0.014 |
| DL      | middle                 |        17 |        10.328 |               7.538 |          0.726 |                  0.005 |
| DL      | fastest third          |        17 |        11.103 |               8.219 |          0.737 |                  0.019 |


No group has a meaningful share of frames at >= 90% of Combine vmax (median 0-0.01%).

![shrinkage](fig/shrinkage.png)
