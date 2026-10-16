# Speed Tax: capacity usage on regular-season routes

_Generated 2026-10-16 by `scripts/speed_tax.py`; window extraction in `src/bdb27/usage.py`._

Each WR/TE route (REG, `route_ran` non-null) is measured from the snap to the throw: peak speed and burst (mean path speed from 2 to 10 yd). Both are expressed as fractions of the player's Combine capacity (`vmax`, `burst2_10`; see `docs/capacity.md`). A linear mixed model with a player random intercept separates situational demand from the player's habitual usage:

- speed: `speedFrac ~ C(routeRan) + bs(tWindow, df=4) + C(align) + C(manZone) + room + motion + C(season) + cushionF + cushionMissing + (1 | player)` (windows ending at a throw, players with at least 40 routes)
- burst: `burstFrac ~ C(routeRan) + C(align) + C(manZone) + cushionF + cushionMissing + C(season) + (1 | player)` (no motion and no speed at the snap)

The **tax** is the player's random intercept, in capacity-fraction units. Negative means he plays below the capacity his peers use in the same situations.

## Coverage

|   REG routes (WR/TE) |   players |   routes with capacity |   players with capacity |   window ends at throw |   sack |   median window (s) |   routes reaching 10 yd of path |
|---------------------:|----------:|-----------------------:|------------------------:|-----------------------:|-------:|--------------------:|--------------------------------:|
|                42851 |       116 |                  36935 |                     103 |                  0.878 |  0.067 |                 2.8 |                           0.636 |


## Capacity used by route (median per route)

| routeRan   |   routes |   medWindow |   medPeak |   speedFrac |   p90speedFrac |   overCapacity |   burstFrac |
|:-----------|---------:|------------:|----------:|------------:|---------------:|---------------:|------------:|
| WHEEL      |       63 |        2.9  |     7.74  |       0.678 |          0.755 |              0 |       0.867 |
| POST       |     3563 |        3    |     7.63  |       0.667 |          0.782 |              0 |       0.89  |
| CROSS      |     4631 |        3.2  |     7.62  |       0.667 |          0.766 |              0 |       0.865 |
| GO         |     6756 |        2.7  |     7.58  |       0.663 |          0.771 |              0 |       0.887 |
| CORNER     |     3056 |        3    |     7.51  |       0.66  |          0.751 |              0 |       0.874 |
| IN         |     3959 |        3    |     6.71  |       0.588 |          0.687 |              0 |       0.844 |
| OUT        |     3514 |        2.7  |     6.34  |       0.557 |          0.66  |              0 |       0.816 |
| FLAT       |     2197 |        2.9  |     5.95  |       0.543 |          0.643 |              0 |       0.754 |
| HITCH      |     5992 |        2.8  |     5.93  |       0.524 |          0.672 |              0 |       0.84  |
| ANGLE      |       30 |        3.45 |     5.315 |       0.497 |          0.596 |              0 |       0.665 |
| SLANT      |     2271 |        2.1  |     5.46  |       0.48  |          0.595 |              0 |       0.713 |
| SCREEN     |      903 |        1.6  |     3.41  |       0.296 |          0.622 |              0 |       0.917 |


## Variance decomposition

| model                          |   routes |   players |   player SD |   residual SD |   ICC per route |   routes for reliability 0.7 |
|:-------------------------------|---------:|----------:|------------:|--------------:|----------------:|-----------------------------:|
| speed used (peak / vmax)       |    32087 |        85 |       0.024 |         0.085 |           0.071 |                       30.372 |
| burst used (2-10 yd / Combine) |    16594 |        78 |       0.036 |         0.089 |           0.138 |                       14.562 |


## Reliability of the tax (split by game parity; year-over-year from per-season fits)

| check                           |   speed tax |   burst tax |
|:--------------------------------|------------:|------------:|
| split-half r (by game)          |       0.873 |       0.917 |
| Spearman-Brown                  |       0.932 |       0.957 |
| players (split-half)            |      77     |      70     |
| year-over-year r (pooled pairs) |       0.873 |       0.9   |
| players (YoY pairs)             |      74     |      64     |


## Route fixed effects in the speed model (relative to the reference route)

| term                  |   coef |
|:----------------------|-------:|
| C(routeRan)[T.SCREEN] | -0.083 |
| C(routeRan)[T.SLANT]  |  0.052 |
| C(routeRan)[T.HITCH]  |  0.056 |
| C(routeRan)[T.FLAT]   |  0.078 |
| C(routeRan)[T.OUT]    |  0.081 |
| C(routeRan)[T.IN]     |  0.093 |
| C(routeRan)[T.WHEEL]  |  0.179 |
| C(routeRan)[T.CORNER] |  0.182 |
| C(routeRan)[T.CROSS]  |  0.183 |
| C(routeRan)[T.GO]     |  0.185 |
| C(routeRan)[T.POST]   |  0.188 |


## Player-level correlations (Spearman)

|           |   speedTax |   burstTax |   vmax |   burst2_10 |   pick |   meanSep |
|:----------|-----------:|-----------:|-------:|------------:|-------:|----------:|
| speedTax  |       1    |       0.27 |  -0.16 |        0.29 |   0.03 |     -0.2  |
| burstTax  |       0.27 |       1    |   0.41 |       -0.42 |   0.1  |     -0.19 |
| vmax      |      -0.16 |       0.41 |   1    |        0.19 |  -0.09 |     -0.06 |
| burst2_10 |       0.29 |      -0.42 |   0.19 |        1    |  -0.21 |     -0.07 |
| pick      |       0.03 |       0.1  |  -0.09 |       -0.21 |   1    |     -0.01 |
| meanSep   |      -0.2  |      -0.19 |  -0.06 |       -0.07 |  -0.01 |      1    |


## Caveats

- **High reliability is expected, not a finding.** In-game speed is a stable player trait, and dividing it by a fixed Combine number keeps it stable. The tax is only interesting if it predicts something (within-player separation; step 3) or differs from absolute game speed.
- **Regression to the mean.** Noise in the Combine denominator shows up as a negative tax–capacity correlation. `vmax` is precise (ICC 0.97), so the speed tax is little affected (ρ = −0.16). The Combine burst is much noisier (within-WR retest ICC only 0.15; the pooled 0.82 overstates it), and burst tax vs `burst2_10` has ρ = −0.42, so the burst tax is largely artefact. Fix in step 3 by shrinking the denominator or modelling log(game) ~ log(Combine) with a slope.
- **Short windows.** Windows end at the throw (median 2.8 s), so deep routes are still accelerating. Speed used means capacity reached by the throw, which mixes acceleration and top speed.
- Raw mean separation is confounded by route mix (deep routes separate less). Compare separation within player and route, not across players.


## Lowest speed tax (plays furthest below capacity)

| name               | pos   |   classYear |   pick |   routes |   vmax |   meanSpeedFrac |   speedTax |   burstTax |
|:-------------------|:------|------------:|-------:|---------:|-------:|----------------:|-----------:|-----------:|
| Darnell Washington | TE    |        2023 |     93 |      444 | 11.307 |           0.475 |     -0.089 |     -0.073 |
| Ryan Flournoy      | WR    |        2024 |    216 |      351 | 12.154 |           0.528 |     -0.053 |      0.012 |
| Cade Stover        | TE    |        2024 |    123 |      189 | 10.922 |           0.524 |     -0.042 |     -0.04  |
| Xavier Legette     | WR    |        2024 |     32 |      784 | 12.196 |           0.552 |     -0.036 |      0.014 |
| Isaiah Bond        | WR    |        2025 |        |      333 | 12.601 |           0.579 |     -0.035 |      0.049 |
| Tip Reiman         | TE    |        2024 |     82 |      128 | 10.81  |           0.529 |     -0.033 |     -0.091 |
| Luther Burden      | WR    |        2025 |     39 |      232 | 11.974 |           0.56  |     -0.029 |      0.042 |
| Blake Whiteheart   | TE    |        2023 |        |      152 | 11.047 |           0.547 |     -0.028 |      0.012 |
| Elijah Higgins     | WR    |        2023 |    197 |      525 | 11.16  |           0.55  |     -0.027 |     -0.056 |
| Rashee Rice        | WR    |        2023 |     55 |      704 | 11.38  |           0.525 |     -0.027 |     -0.043 |


## Highest speed tax

| name              | pos   |   classYear |   pick |   routes |   vmax |   meanSpeedFrac |   speedTax |   burstTax |
|:------------------|:------|------------:|-------:|---------:|-------:|----------------:|-----------:|-----------:|
| Ronnie Bell       | WR    |        2023 |    253 |      152 | 10.913 |           0.667 |      0.054 |      0.012 |
| A.T. Perry        | WR    |        2023 |    195 |      192 | 11.129 |           0.646 |      0.041 |      0.01  |
| Jalen Brooks      | WR    |        2023 |    244 |      302 | 10.971 |           0.63  |      0.04  |      0.044 |
| Malik Heath       | WR    |        2023 |        |      266 | 10.828 |           0.631 |      0.037 |     -0.014 |
| Xavier Hutchinson | WR    |        2023 |    205 |      794 | 11.157 |           0.629 |      0.032 |      0.023 |
| Ladd McConkey     | WR    |        2024 |     34 |     1007 | 10.855 |           0.61  |      0.029 |     -0.078 |
| Pat Bryant        | WR    |        2025 |     74 |      282 | 11.127 |           0.612 |      0.028 |      0.039 |
| Jermaine Burton   | WR    |        2024 |     80 |       95 | 11.668 |           0.665 |      0.026 |      0.006 |
| Konata Mumpfield  | WR    |        2025 |    242 |      168 | 10.855 |           0.63  |      0.025 |      0.016 |
| Rome Odunze       | WR    |        2024 |      9 |      928 | 11.136 |           0.627 |      0.025 |     -0.024 |
