# Forty Cents on the Dollar: How Much Combine Speed Survives to Sunday

*NFL Big Data Bowl 2027, Combine to NFL Performance*

![hero](../fig/hero.gif)
*Figure 1. [TBD caption: two receivers' Combine 40 speed traces (left) beside their fastest regular-season vertical reps (right).]*

## The question every scout already asks

A receiver runs 4.35. Another runs 4.45. Every draft room hears that tenth of a second and wonders how much of
it will show up on Sundays. Until now, nobody could check. The stopwatch ends at 40 yards, and game speed was
measured on a different system for different players.

The 2027 data closes that loop for the first time: the same 510 prospects were tracked at 10 Hz both at the
Combine and in every regular-season snap of 2023–2025. We use it to measure a **translation rate**: for every
1% of Combine top speed a prospect has over a peer at his position, how much faster is he in NFL games?

**The answer is about 0.4%. Within a position, a Combine speed edge is worth roughly forty cents on the dollar.**

## Data and approach

**Combine capacity (tracking, not stopwatch).** We fit each tracked 40-yard dash (793 attempts by 418
prospects) with the sprint model used in sports science, v(t) = v<sub>max</sub>(1 − e<sup>−(t−t₀)/τ</sup>).
The start time t₀ is a free parameter, so inconsistently trimmed clips do not bias the fit (median R² 0.995).
Top speed v<sub>max</sub> extrapolates beyond the end of the clip, which matters because 97% of clips end
with the runner still at speed. The curve-implied 40 matches the official time (r = 0.97, with a
constant +0.25 s offset from the start trigger). v<sub>max</sub> repeats rep to rep with ICC 0.80–0.97
within every position group.

**Game speed.** For each regular-season snap we measure the player's peak speed from the snap to the throw
(or sack): 35k WR routes, 35k DB coverage snaps on thrown passes, and 33k DL pass rushes. A mixed model
removes situational demand (route or coverage assignment, alignment, time to throw, down and distance, field
position, cushion, season), leaving each player's game-speed level.

**Translation rate.** We regress log game speed on log Combine top speed within each position group,
controlling for body weight and the context above, with a player random effect. The slope is an
elasticity: 1 means Combine differences show up in full, and 0 means they vanish.

![translation](../fig/translation.png)
*Figure 2. Combine-to-game elasticity of top speed by position group, under three definitions of game speed.
Points are estimates and bars are 95% CIs.*

## Result 1: speed edges shrink by more than half

| Group | All reps (snap → throw) | Long reps (≥ 3.5 s) | Whole play, fastest 0.1% |
|---|---|---|---|
| WR | 0.27 [0.07, 0.47] | 0.38 [0.13, 0.62] | 0.29 [0.01, 0.57] |
| DB | 0.38 [0.02, 0.73] | 0.47 [0.01, 0.93] | 0.38 [0.02, 0.74] |
| DL | 0.42 [0.08, 0.77] | 0.45 [0.09, 0.81] | 0.62 [0.27, 0.97] |

*Table 1. Elasticities, weight-controlled, within position.*

- **It is not a short-route artifact.** Restricting to long vertical routes, or to each player's fastest
  0.1% of frames anywhere in a play (after the catch, in pursuit), gives the same range. At that extreme,
  receivers reach about 84% of their Combine top speed.
- **It is not measurement noise.** Combine top speed is highly repeatable within groups. Correcting for its
  error moves WR from 0.27 to about 0.31.
- **Pooling hides it.** Pool WR with TE, or ignore DL body weight, and transfer looks complete (0.82–1.0). That
  pooled number measures body type, not translation.

## Result 2: what explains the shrinkage

[TBD from docs/shrinkage.md: role, draft capital, experience, usage, fatigue moderators. 1 figure + 2-3 bullets.]

## Result 3: where speed pays, within the same player

Shrunk or not, speed is valuable. Comparing a player's faster and slower reps on the same route type removes
talent and draft-slot confounding:

- **WR vertical routes (go, post, corner, wheel):** +0.10 yd separation at the throw per SD of rep speed
  [0.06, 0.15].
- **WR breaking routes (in, out, cross):** no separation change, but +2.8 pp target probability per SD
  [1.7, 3.9].
- **WR short routes:** faster reps show *less* separation, because receivers who settle into soft spots are
  slower and open. This is not a cost of speed.
- **DL:** a quicker get-off (NGS, first steps before contact) adds +3.2 pp pressure probability per SD on the
  edge and +2.6 pp inside.

## Sunday Speed: a projection scouts can use

[TBD from docs/sunday_speed.md: model, leave-one-class-out validation vs naive full transfer and vs the
official forty, the 2025-class scout table, and "tests fast, plays slower" examples. 1 figure + 1 table.]

## What the tracking adds, and what it doesn't

We tested whether tracking-derived capacity (top speed, 2–10 yd burst) predicts game speed, separation over
expected, target rate or pressure *beyond* draft slot, the official forty and size. With 60–85 players per
group, it does not: every ΔR² is within ±0.03 of zero. The tracked 40 is not a better stopwatch. Its value is
that it is **on the same scale as game tracking**, which is what makes a translation rate measurable at all.
We also report what did not survive:
- the 2–10 yd "burst" is unreliable within receivers (retest ICC 0.15);
- skill-drill change-of-direction features do not repeat across drills;
- the share of capacity used does not predict separation better than raw speed.

## How a team would use this

- **Draft season.** Discount 40-time gaps between same-position prospects by about 60% when projecting game
  speed. Speed edges among receivers matter on vertical stems, so put the premium there.
- **In season, week to week.** Track each player's game top speed against his own Sunday Speed projection. A
  sustained drop below projection is a monitoring flag for soft-tissue or return-to-play cases.
- **Development.** For DL, get-off is the speed lever that pays (pressure per SD of quicker first step).

## Limitations

- **Selection.** Only players who earn snaps have game speed. Draft slot predicts playing time (R² 0.37–0.64),
  and 40–45% of each group lacks a usable per-snap sample.
- **Exposure.** The 2025 class has one season.
- **Missing tracked 40s.** About 18% of prospects lack one.
- **No opponents.** Game tracking covers only the cohort, so speed is never measured against a specific
  defender.
- **Small samples.** Elasticity CIs are wide for DB.

## Appendix

Code: [public Kaggle notebook link TBD]. Repository modules: `bdb27.capacity` (sprint fits), `bdb27.usage`
(game windows), `scripts/translation.py` (elasticities), `scripts/shrinkage.py`, `scripts/sunday_speed.py`.
