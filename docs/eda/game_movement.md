# Game-movement mirror metrics (drill-to-game kinematic bridge)

**Question.** Which in-game *movement* metrics can we measure reliably from the cohort's own REG-season tracking, as mirrors of the Combine drills? And do the mirrors actually line up with the drills?

**Code.** `scripts/eda/game_movement.py` (`PYTHONPATH=src python scripts/eda/game_movement.py [--refresh]`).
- Play-level caches are written to `data/processed/cache/gm_{speed,breaks,trigger,dl_first,ol_set}.parquet`.
- All numbers are in `data/processed/cache/gm_results.json`.
- Each season is scanned lazily with column and row filters. Peak RSS was about 1.3 GB.

**Conventions.**
- Data is REG only.
- **Reliability** is split-half: odd/even plays per player in time order, Pearson r across players, then Spearman-Brown (**SB**). `rel@n` projects the single-half reliability to the median play count of *all* players with any plays. It is lower than SB when many players have few snaps.
- Metrics are z-scored within context (route, coverage family, alignment) before players are averaged.
- Correlations are Spearman with Fisher-z 95% CIs.

## 1. Speed Tax: in-game top speed vs Combine 40 peak speed

**Data.** 251,081 player-plays, 18.1M frames, 401 players. Play max `s` is capped at 12.5 yd/s (8 teleports dropped). The player-level table uses players with 50 or more plays. Combine peak speed is the maximum tracked `s` over FORTY_YARD_DASH attempts (418 players).

| Group | n (with 40 trk) | Plays (median) | Combine 40 peak | Game p99 of play-max | Share p99 / max | Game max > 40 peak | r(p99, 40 peak) | r(p99, 40 time) |
|---|---|---|---|---|---|---|---|---|
| WR | 70 (64) | 352 | 11.15 | 9.53 | 0.86 / 0.90 | 0% | **+0.65** [+0.47, +0.77] | −0.56 |
| TE | 23 (20) | 262 | 10.48 | 8.54 | 0.82 / 0.86 | 5% | +0.57 [+0.16, +0.81] | −0.55 |
| DB | 79 (68) | 620 | 11.04 | 9.27 | 0.84 / 0.92 | 4% | +0.32 [+0.08, +0.52] | −0.33 |
| DL | 88 (70) | 369 | 10.12 | 7.34 | 0.74 / 0.82 | 0% | **+0.72** [+0.58, +0.82] | −0.77 |
| OL | 77 (63) | 982 | 9.48 | 6.14 | 0.65 / 0.79 | 0% | +0.31 [+0.06, +0.52] | −0.24 |

**Correlations.**
- Pooled across groups: r = +0.88 (n = 285). That is mostly between-group differences.
- Within group (z-scored): **r = +0.51** [+0.41, +0.59].
- Game p95 peak acceleration vs Combine 40 peak acceleration, within group: only r = +0.20 [+0.09, +0.32].

**Split-half reliability (SB).**

| Group | p95 play-max speed | p99 | max | p95 max accel |
|---|---|---|---|---|
| WR (n = 70) | 0.94 | 0.82 | 0.75 | 0.72 |
| TE (n = 23) | 0.82 | 0.71 | 0.62 | 0.75 |
| DB (n = 79) | 0.50 | 0.67 | 0.71 | 0.82 |
| DL (n = 88) | 0.92 | 0.88 | 0.84 | 0.97 |
| OL (n = 77) | 0.69 | 0.49 | 0.76 | 0.86 |

**Takeaways.**
- Rookies essentially never exceed their Combine top speed in a game.
- WR/DB use about 85–92% of it, DL about 75–82% and OL about 65–80%.
- In-game top speed is reliable for WR and DL, and it tracks Combine speed for WR, TE and DL.
- For DB and OL it is weak (r ≈ 0.3), because role decides how often they get to open-field sprints.

## 2. Brake Check: WR/TE route breaks vs Combine route-drill cuts vs separation

**Data.**
- 22,738 REG route plays (HITCH/OUT/IN/SLANT/POST/CORNER; `route_ran` has no CURL/COMEBACK labels) from 116 WR/TE.
- 907k frames, covering the snap to `pass_forward` + 1 s.
- Plays without a `pass_forward` (sacks, runs) are dropped.

**Detector.**
- `detect_cuts` as written (speed-minimum prominence of 0.75 yd/s, turn of 25° or more) finds a break before the throw in only **14%** of plays (CORNER/POST 5%). Rounded game breaks rarely show a prominent speed dip.
- I therefore added a heading-based `route_break()` to the script. It takes the first peak of |Δheading over 0.8 s| of 35° or more between snap + 0.5 s and `pass_forward`. It then measures:
  - entry speed: the maximum in the 1 s before the break
  - minimum speed at the break
  - speed retained
  - peak and mean deceleration
  - deceleration time
  - peak re-acceleration (1 s after)
  - turn angle
- Overall detection is 73%, ranging from POST 50% and CORNER 56% to IN 89% and OUT 93%.
- Route medians are sensible:
  - HITCH: 4.8 → 1.6 yd/s, peak decel 4.6
  - OUT: 5.5 → 4.9
  - POST and CORNER keep about 95–99% of their speed, which is effectively no brake.
- The same detector is applied to the Combine drills: 547 attempts from 149 players. CURL drops 6.8 → 1.9 yd/s (peak decel 6.8). SPEED_OUT and DAGGER barely decelerate.

**Reliability (route-z, 86 players with 15 or more breaks per half, about 70 per half).**
- Strong: entry speed SB 0.93, minimum speed 0.83, deceleration time 0.77, re-acceleration 0.77.
- Moderate: peak/mean deceleration 0.72.
- Weak: speed retained 0.52, turn angle 0.44, COD time 0.38.
- On hard breaks only (HITCH/OUT/IN, 81 players) the picture is the same: deceleration 0.72, speed retained 0.49.

**Combine side.**
- Same-drill retest is decent (n = 15): entry speed r = 0.82, speed retained 0.69, peak decel 0.67.
- The cross-drill trait is absent. Curl+comeback vs speed-out+dagger z-scores give r = +0.12 for peak decel and +0.07 for speed retained (n = 103). Combine "cut quality" is drill-specific, not a stable trait.

**Combine vs game (WR, n = 65; 30 or more game breaks).**
- Every metric is null: |r| ≤ 0.16, and every CI spans 0.
- Examples: peak decel +0.05 [−0.20, +0.30], speed retained −0.05, re-acceleration +0.16.
- Hard-break-only results are also null (n = 62).
- The earlier strict `detect_cuts` version (n = 28) gave peak decel +0.36 [−0.02, +0.66]. With so few players that is noise-level.

**Separation (route-z separation_at_pass_forward; SB 0.72, 92 players).**
- **Player level** (n = 86): entry speed **+0.28** [+0.07, +0.47] and minimum speed +0.26. Deceleration +0.15 is not significant, and speed retained is −0.09.
- **Play level, within player** (n = 16,501): entry speed +0.10, minimum speed +0.14, peak decel −0.01, re-acceleration −0.12. Faster breaks happen on plays with more separation, but a harder brake does not.
- **Combine cut metrics vs game separation**: all null (|r| ≤ 0.14, n = 83).

**Takeaway.** In-game break *speed* (entry and minimum) is a reliable trait and weakly tracks separation. Brake *quality* (deceleration, speed retained) is only moderately reliable. It has no measurable link to the Combine route drills or to separation.

## 3. Trigger Time: DB reaction after pass_forward vs Combine back-pedal 45° drill

**Data.**
- 33,926 REG DB coverage player-plays (MAN 8.9k, deep zone 15.9k, underneath 9.1k), 92 players.
- 1.50M frames, from the snap to `pass_forward` + 1.5 s.
- `response_after_event` with a 1.5 s window.

**Distributions.**
- Median time to peak acceleration is 0.4 s. In 32% of plays the peak is at frame 0, meaning the DB was already accelerating.
- 48% of plays never gain 1 yd/s within 1.5 s.
- Median heading change is 56°.
- The metric is dominated by role and leverage, not by reaction.

**Reliability.**
- Coverage-z (74 players, about 165 plays per half):
  - peak accel SB 0.85
  - |Δheading| 0.76
  - speed gain in 1 s 0.72
  - tPeakAccel 0.70
  - tSpeedGain 0.54
- Residualized on coverage family, depth and speed at the throw: tPeakAccel 0.64, speed gain 0.69, |Δheading| 0.71, peak accel 0.81.
- MAN only: tPeakAccel 0.37 and tSpeedGain 0.39, which is weak.

**Combine side.**
- The 45° transition is detected in only 102/241 attempts (64 players). The drill has no stimulus timestamp, so true reaction cannot be measured; only the transition kinematics can.
- Attempt retest of transition COD time: r = −0.01 [−0.35, +0.33] (n = 34). It is unreliable.
- Game vs Combine (n = 37–38): all null. The best is tPeakAccel vs time to regain, r = −0.26 [−0.54, +0.09].

**Takeaway.** A game "trigger" is measurable and fairly reliable once residualized. But without the ball or the receiver, it mixes reaction with assignment, and the Combine counterpart is noise. There is no bridge.

## 4. DL first step vs NGS player_get_off

**Data.**
- 52,584 REG DL snaps (841k frames, snap to +1.5 s). 33,292 of them carry `player_get_off`, from 98 players.
- Measures are z-scored within EDGE vs interior line.

**Get-off is a re-measure of the tracking burst.**
- Play level: get-off vs snap-relative x displacement at +1.0 s r = **+0.90**, distance at +1.0 s −0.77, speed at +0.5 s −0.75, time to 3 yd/s +0.70.
- Player level (n = 77): x displacement at +1.0 s r = +0.87 [+0.80, +0.92], distance −0.85.
- Within player: −0.75.

**Reliability (77 players).** All measures are excellent:
- distance at +0.5 s SB 0.97, at +1.0 s 0.96
- speed at +0.5 s 0.97, at +1.0 s 0.94
- time to 3 yd/s 0.92
- NGS get-off 0.94

**Combine links.**
- 10-yd split vs distance at +1.0 s: r = −0.34 [−0.55, −0.09]; vs speed at +1.0 s −0.39; vs get-off +0.36 (n = 62).
- 40 peak acceleration: +0.37.
- **PASS_RUSH_DRILL burst** (from motion onset: distance and speed at +0.5/1.0 s, time to 3 yd/s): r ≈ 0 for every pairing (|r| ≤ 0.14, n = 77). Motion onset in an untimed drill is not a snap.

**Takeaway.** Use NGS get-off directly; it already captures the burst. The only drill bridge is the stopwatch 10-yd split or 40 acceleration (r ≈ 0.35). The Combine pass-rush drill adds nothing.

## 5. OL pass set

**Data.**
- 48,536 REG pass-block snaps (`pressure_allowed` non-null) from 88 OL lined up at T, G or C. 1.38M frames.
- Window is the first 1.0 s after the snap.
- Measures:
  - maximum and mean backward velocity (−vx)
  - maximum |vy|
  - set depth and set width at +1.0 s (width positive away from the center)
  - total set distance
- Tackles set deepest: max back velocity 2.1 yd/s, depth 1.26 yd, width +0.52. Guards: 1.45 yd/s, 0.72 yd.
- Measures are z-scored within T/G/C.

**Reliability (69 players, 30 or more per half).**
- Set metrics are excellent: max back velocity SB 0.94, mean 0.94, depth 0.94, distance 0.95, lateral velocity 0.93. Set width is 0.58.
- Outcomes are moderate: pressure rate 0.64, peak pressure probability 0.73.

**Pressure (player level, n = 66).**
- Deeper and faster backward sets go with *more* pressure: max back velocity vs peak pressure probability r = +0.60 [+0.42, +0.74], depth +0.64, lateral velocity −0.56.
- The same holds within player and play level: +0.41. After controlling dropback duration it is still +0.37. Even depth at +0.5 s gives +0.31.
- Dropback duration itself is unrelated to set depth (r = 0.01).
- So this reflects the *rush* (speed rushers force deeper, faster sets, bull rushes push the blocker back), not blocker skill. It is confounded and probably reverse-causal. Without opponents in the data we cannot separate set from push.

**Combine links (n = 66).** Every link with the PASS_PRO_MIRROR and five-yard wave drills (peak speed, accel, decel) or with 40 acceleration is null (|r| ≤ 0.23). The Combine OL drills also show no link to pressure (|r| ≤ 0.12).

**Takeaway.** The pass-set kinematics are among the most reliable game metrics we have. However, they have no Combine counterpart, and their pressure association is contaminated by the unseen rusher.

## Implications for project ideas

| Idea | Game mirror reliable? | Drill ↔ game link | Bridge rating |
|---|---|---|---|
| **Speed Tax** (Combine top speed → game top speed) | Yes. WR/DL SB 0.9; DB/OL 0.5–0.7 | WR +0.65, DL +0.72, TE +0.57; DB/OL +0.3; within-group pooled +0.51 | **Strong** (WR, TE, DL); ok for DB/OL |
| **DL first step / get-off** | Yes, SB ≥ 0.92; NGS get-off ≈ the same quantity (r 0.87–0.90) | 10-yd split / 40 acceleration r ≈ 0.35; PASS_RUSH_DRILL burst ≈ 0 | **Ok**, via stopwatch splits only. Use NGS get-off as the outcome |
| **Brake Check** (route breaks) | Entry/minimum speed yes (SB 0.83–0.93); deceleration moderate (0.72); speed retained weak (0.5) | Combine cut metrics vs game breaks ≈ 0; Combine cuts not even consistent across drills; break deceleration does not predict separation (entry speed +0.28 does) | **Weak** |
| **Trigger Time** (DB reaction) | Moderate (residualized SB 0.64–0.81), but role-driven; MAN only 0.37 | Combine BP45 transition unreliable (retest ≈ 0) and unrelated to game (|r| ≤ 0.26, n = 37) | **Weak** |
| **OL pass set / Mirror** | Very reliable (SB 0.94) | PASS_PRO_MIRROR / wave ≈ 0; pressure association is confounded by the unseen rusher | **Weak** |

**Practical guidance.**
1. Only straight-line speed and acceleration survive the trip from drill to game. The project's kinematic bridge should be built on Speed Tax: share of Combine speed used, by group. For DL, also include burst, with NGS get-off as the game-side measure.
2. For WR, the reliable game-side movement signals are break entry speed and minimum speed. Use them as *outcomes* or mediators toward separation, not as mirrors of Combine route drills.
3. Combine COD/route/reaction drill kinematics are drill-specific and low-retest (no stimulus timing, mostly single reps). Any idea that claims a drill-to-game "agility" bridge should expect r ≈ 0 at n ≈ 60–80 and needs to be framed accordingly.

**Caveats.**
- The `route_break` detector is new and prototype-level. It misses about half of the POST/CORNER breaks.
- Game breaks are measured only up to `pass_forward`.
- n for Combine comparisons is 37–77 players, so CIs are wide (±0.25). Small true effects (r ≈ 0.2) cannot be excluded.
