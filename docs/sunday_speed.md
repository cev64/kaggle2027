# Sunday Speed: projecting NFL game top speed from the Combine

_Generated 2026-10-16 by `scripts/sunday_speed.py`. Per-player table: `data/processed/sunday_speed_players.parquet`._

## Summary

- **Target: whole-play p99 speed** (fastest 1% of a player's REG frames, all plays, players with >= 200 plays).
  Split-half reliability (alternating games) is 0.56-0.93,
  versus 0.74-0.95
  for the context-adjusted route/rush-window intercept, and it reads directly as a top speed in yd/s.
  The two targets agree across players (r = WR 0.72, DB 0.18, DL 0.69).
- **Shrunk beats naive full transfer by a clear margin.** Leave-one-draft-class-out RMSE, all groups:
  naive 0.378 yd/s, shrunk 0.316 yd/s
  (difference 0.062, 95% CI 0.032 to 0.092; 16% lower).
  The naive calibration slope is 0.94: it spreads players out about
  1.1x too much. It is worse than the plain group mean. Shrunk slope: 0.98.
- **Estimated elasticities on p99 (all data, weight held fixed):** WR 0.43, DB 0.36, DL 0.10. These match
  "forty cents on the dollar".
- **Stopwatch equivalent: yes.** Official forty + weight is as good as tracking vmax + weight (RMSE difference
  -0.006 yd/s, 95% CI -0.019 to 0.007), and adding vmax to the forty does not help.
  The tracked 40 is a convenience, not an edge, for projecting Sunday speed.
- **Honest ceiling.** Even the best model explains only part of the spread. The 80% interval is about
  ±0.7 mph, and held-out coverage of the 80% intervals is 81%
  (WR 81%, DB 82%, DL 80%).

## How a scout uses this

Start from the prospect's tracked 40 top speed (or official forty) and weight, and read off the **projected Sunday
speed with its 80% band** instead of the raw Combine number. Combine differences survive at roughly 40 cents on the
dollar, so a receiver who is 1 mph faster than his group at the Combine projects only about
0.4 mph faster on Sundays. Two players whose bands overlap should be treated as the same speed for game
planning. Use the band, not the point, when the gap between two prospects is under about 1 mph. The "naive" column is
what you implicitly assume when you trust the stopwatch at face value. Its largest misses (below) are
"tests fast, plays slower" players: the projection had already pulled them back toward the group.
The separation column is a rough illustration only (see its caveat). It is not a validated between-player effect.

## 1. Choosing the game-speed target (split-half by alternating games, Spearman-Brown)

Whole-play halves need >= 100 plays per half. Window-intercept halves need >= 20 windows per half and refit the context + window-length mixed model of `scripts/translation.py` on each half.

| group   | target                                     |   players |   split-half r |   Spearman-Brown reliability |
|:--------|:-------------------------------------------|----------:|---------------:|-----------------------------:|
| WR      | window random intercept (context-adjusted) |        63 |          0.8   |                        0.889 |
| WR      | whole-play p99                             |        49 |          0.864 |                        0.927 |
| WR      | whole-play p99.9                           |        49 |          0.628 |                        0.771 |
| DB      | window random intercept (context-adjusted) |        62 |          0.591 |                        0.743 |
| DB      | whole-play p99                             |        64 |          0.393 |                        0.564 |
| DB      | whole-play p99.9                           |        64 |          0.429 |                        0.6   |
| DL      | window random intercept (context-adjusted) |        60 |          0.898 |                        0.946 |
| DL      | whole-play p99                             |        63 |          0.858 |                        0.923 |
| DL      | whole-play p99.9                           |        63 |          0.722 |                        0.838 |


Modeling sample (tracked 40, official forty, weight, >= 200 REG plays): WR 48, DB 55, DL 51 players.

## 2. Out-of-sample projection accuracy (yd/s)

Models are fit on the log scale within group. (a) uses the training-group mean ratio of game to Combine vmax (elasticity 1); (b) is the training-group mean; (c0)-(e) are OLS on the log predictors. Calibration slope is the slope of observed on predicted (1 = ideal; < 1 = predictions too spread). It is undefined for the group mean.

### Leave-one-draft-class-out (train on two classes, predict the third)

| group   | model                       |   n |   RMSE |   MAE |   calibration slope |
|:--------|:----------------------------|----:|-------:|------:|--------------------:|
| WR      | (a) naive full transfer     |  48 |  0.324 | 0.261 |               0.462 |
| WR      | (b) group mean              |  48 |  0.331 | 0.283 |              -2.11  |
| WR      | (c0) elasticity only        |  48 |  0.284 | 0.223 |               0.788 |
| WR      | (c) shrunk: vmax + weight   |  48 |  0.242 | 0.199 |               0.928 |
| WR      | (d) official forty + weight |  48 |  0.262 | 0.215 |               0.794 |
| WR      | (e) vmax + forty + weight   |  48 |  0.249 | 0.204 |               0.837 |
| DB      | (a) naive full transfer     |  55 |  0.346 | 0.286 |               0.276 |
| DB      | (b) group mean              |  55 |  0.286 | 0.223 |              -2.021 |
| DB      | (c0) elasticity only        |  55 |  0.281 | 0.218 |               0.574 |
| DB      | (c) shrunk: vmax + weight   |  55 |  0.281 | 0.219 |               0.551 |
| DB      | (d) official forty + weight |  55 |  0.269 | 0.208 |               0.729 |
| DB      | (e) vmax + forty + weight   |  55 |  0.273 | 0.21  |               0.638 |
| DL      | (a) naive full transfer     |  51 |  0.452 | 0.363 |               0.666 |
| DL      | (b) group mean              |  51 |  0.536 | 0.449 |              -3.038 |
| DL      | (c0) elasticity only        |  51 |  0.435 | 0.356 |               0.97  |
| DL      | (c) shrunk: vmax + weight   |  51 |  0.402 | 0.331 |               0.958 |
| DL      | (d) official forty + weight |  51 |  0.384 | 0.317 |               1.028 |
| DL      | (e) vmax + forty + weight   |  51 |  0.386 | 0.31  |               0.976 |
| all     | (a) naive full transfer     | 154 |  0.378 | 0.304 |               0.941 |
| all     | (b) group mean              | 154 |  0.398 | 0.316 |               0.981 |
| all     | (c0) elasticity only        | 154 |  0.341 | 0.265 |               0.982 |
| all     | (c) shrunk: vmax + weight   | 154 |  0.316 | 0.25  |               0.983 |
| all     | (d) official forty + weight | 154 |  0.31  | 0.246 |               0.987 |
| all     | (e) vmax + forty + weight   | 154 |  0.309 | 0.242 |               0.989 |


### Repeated 5-fold CV (40 repeats, mean)

| group   | model                       |   RMSE |   MAE |   calibration slope |
|:--------|:----------------------------|-------:|------:|--------------------:|
| DB      | (a) naive full transfer     |  0.322 | 0.262 |               0.338 |
| DB      | (b) group mean              |  0.291 | 0.227 |              -4     |
| DB      | (c0) elasticity only        |  0.281 | 0.218 |               0.61  |
| DB      | (c) shrunk: vmax + weight   |  0.272 | 0.213 |               0.717 |
| DB      | (d) official forty + weight |  0.269 | 0.212 |               0.746 |
| DB      | (e) vmax + forty + weight   |  0.272 | 0.216 |               0.676 |
| DL      | (a) naive full transfer     |  0.447 | 0.365 |               0.675 |
| DL      | (b) group mean              |  0.531 | 0.443 |              -3.99  |
| DL      | (c0) elasticity only        |  0.435 | 0.359 |               0.923 |
| DL      | (c) shrunk: vmax + weight   |  0.38  | 0.302 |               0.954 |
| DL      | (d) official forty + weight |  0.374 | 0.302 |               0.961 |
| DL      | (e) vmax + forty + weight   |  0.381 | 0.303 |               0.926 |
| WR      | (a) naive full transfer     |  0.315 | 0.256 |               0.486 |
| WR      | (b) group mean              |  0.318 | 0.27  |              -4.017 |
| WR      | (c0) elasticity only        |  0.278 | 0.218 |               0.866 |
| WR      | (c) shrunk: vmax + weight   |  0.239 | 0.195 |               0.934 |
| WR      | (d) official forty + weight |  0.247 | 0.205 |               0.906 |
| WR      | (e) vmax + forty + weight   |  0.243 | 0.199 |               0.892 |


### Paired contrasts (LODO predictions, bootstrap over players)

| group   | RMSE(A) - RMSE(B)                                       |   yd/s |   lo95 |   hi95 |   % of A |
|:--------|:--------------------------------------------------------|-------:|-------:|-------:|---------:|
| WR      | (a) naive full transfer - (c) shrunk: vmax + weight     |  0.082 |  0.028 |  0.137 |   25.247 |
| WR      | (b) group mean - (c) shrunk: vmax + weight              |  0.089 |  0.039 |  0.141 |   26.815 |
| WR      | (d) official forty + weight - (c) shrunk: vmax + weight |  0.02  | -0.01  |  0.051 |    7.761 |
| WR      | (d) official forty + weight - (e) vmax + forty + weight |  0.013 | -0.003 |  0.029 |    5.11  |
| DB      | (a) naive full transfer - (c) shrunk: vmax + weight     |  0.065 |  0.022 |  0.104 |   18.765 |
| DB      | (b) group mean - (c) shrunk: vmax + weight              |  0.005 | -0.031 |  0.041 |    1.67  |
| DB      | (d) official forty + weight - (c) shrunk: vmax + weight | -0.012 | -0.03  |  0.006 |   -4.538 |
| DB      | (d) official forty + weight - (e) vmax + forty + weight | -0.005 | -0.014 |  0.004 |   -1.792 |
| DL      | (a) naive full transfer - (c) shrunk: vmax + weight     |  0.05  | -0.009 |  0.109 |   11.113 |
| DL      | (b) group mean - (c) shrunk: vmax + weight              |  0.135 |  0.069 |  0.201 |   25.118 |
| DL      | (d) official forty + weight - (c) shrunk: vmax + weight | -0.018 | -0.037 |  0     |   -4.687 |
| DL      | (d) official forty + weight - (e) vmax + forty + weight | -0.003 | -0.032 |  0.023 |   -0.686 |
| all     | (a) naive full transfer - (c) shrunk: vmax + weight     |  0.062 |  0.032 |  0.092 |   16.433 |
| all     | (b) group mean - (c) shrunk: vmax + weight              |  0.082 |  0.048 |  0.117 |   20.628 |
| all     | (d) official forty + weight - (c) shrunk: vmax + weight | -0.006 | -0.019 |  0.007 |   -1.994 |
| all     | (d) official forty + weight - (e) vmax + forty + weight |  0.001 | -0.013 |  0.013 |    0.279 |


![sunday speed](fig/sunday_speed.png)

## 3. 2025 class: held-out projections vs actual (trained on 2023-24 only; first 8 picks per group)

| name                  | group   |   pick |   forty |   vmax (mph) |   naive (mph) | Sunday proj (mph) [80%]   |   actual (mph) |   miss shrunk (mph) |
|:----------------------|:--------|-------:|--------:|-------------:|--------------:|:--------------------------|---------------:|--------------------:|
| Malaki Starks         | DB      |     27 |    4.5  |         24.2 |          17.7 | 17.3 [16.6, 18.0]         |           16.7 |               -0.56 |
| Maxwell Hairston      | DB      |     30 |    4.28 |         25.2 |          18.4 | 17.9 [17.1, 18.7]         |           18.2 |                0.34 |
| Trey Amos             | DB      |     61 |    4.43 |         23.5 |          17.1 | 17.1 [16.4, 17.8]         |           17.2 |                0.13 |
| Darien Porter         | DB      |     68 |    4.3  |         24.9 |          18.1 | 17.5 [16.8, 18.3]         |           16.9 |               -0.69 |
| Nohl Williams         | DB      |     85 |    4.5  |         24.2 |          17.7 | 17.3 [16.6, 18.0]         |           17.2 |               -0.07 |
| Jonas Sanker          | DB      |     93 |    4.48 |         24.2 |          17.6 | 17.1 [16.4, 17.8]         |           17.1 |                0.04 |
| Upton Stout           | DB      |    100 |    4.44 |         23.7 |          17.3 | 17.4 [16.7, 18.2]         |           17.1 |               -0.33 |
| Craig Woodson         | DB      |    106 |    4.45 |         23.5 |          17.2 | 17.0 [16.3, 17.7]         |           16   |               -0.96 |
| Darius Alexander      | DL      |     65 |    4.95 |         20.6 |          13.2 | 12.9 [11.9, 13.9]         |           12.7 |               -0.16 |
| Princely Umanmielen   | DL      |     77 |    4.72 |         21.8 |          13.9 | 14.5 [13.4, 15.7]         |           14.5 |               -0.02 |
| Jordan Burch          | DL      |     78 |    4.67 |         23.4 |          14.9 | 13.8 [12.7, 14.9]         |           13.1 |               -0.67 |
| C.J. West             | DL      |    113 |    4.95 |         20.8 |          13.3 | 12.7 [11.7, 13.7]         |           12.4 |               -0.26 |
| Joshua Farmer         | DL      |    137 |    5.11 |         20   |          12.8 | 12.8 [11.9, 13.8]         |           12.6 |               -0.25 |
| Tyrion Ingram-Dawkins | DL      |    139 |    4.86 |         21.4 |          13.6 | 13.6 [12.6, 14.7]         |           11.8 |               -1.77 |
| Elijah Roberts        | DL      |    157 |    4.78 |         21.9 |          14   | 13.5 [12.4, 14.5]         |           13   |               -0.43 |
| Yahya Black           | DL      |    164 |    5.39 |         18.6 |          11.9 | 12.1 [11.1, 13.1]         |           12.3 |                0.22 |
| Jayden Higgins        | WR      |     34 |    4.47 |         23.1 |          18   | 17.8 [17.2, 18.5]         |           18.2 |                0.42 |
| Luther Burden         | WR      |     39 |    4.41 |         24.5 |          19.1 | 18.3 [17.7, 19.0]         |           18.5 |                0.13 |
| Isaac TeSlaa          | WR      |     70 |    4.43 |         24.3 |          18.9 | 18.2 [17.5, 18.9]         |           18.7 |                0.58 |
| Pat Bryant            | WR      |     74 |    4.61 |         22.8 |          17.7 | 17.9 [17.2, 18.5]         |           18   |                0.14 |
| Jaylin Noel           | WR      |     79 |    4.39 |         24   |          18.7 | 18.4 [17.8, 19.1]         |           18.5 |                0.06 |
| Chimere Dike          | WR      |    103 |    4.34 |         24.2 |          18.8 | 18.4 [17.8, 19.1]         |           18.4 |               -0.02 |
| Dont'e Thornton       | WR      |    108 |    4.3  |         24.6 |          19.2 | 18.4 [17.7, 19.1]         |           18.8 |                0.38 |
| Arian Smith           | WR      |    110 |    4.36 |         24.5 |          19.1 | 18.9 [18.2, 19.6]         |           19.5 |                0.59 |


## 4. Biggest misses of naive full transfer (LODO, all classes)

**Tests fast, plays slower** (naive projection far above the game):

| name                  | group   |   pick |   forty |   vmax (mph) |   naive (mph) | Sunday proj (mph) [80%]   |   actual (mph) |   draftYear |   miss naive (mph) |   miss shrunk (mph) |
|:----------------------|:--------|-------:|--------:|-------------:|--------------:|:--------------------------|---------------:|------------:|-------------------:|--------------------:|
| Ryan Flournoy         | WR      |    216 |    4.44 |         24.9 |          19.4 | 18.7 [18.0, 19.3]         |           17.5 |        2024 |              -1.92 |               -1.19 |
| Jordan Burch          | DL      |     78 |    4.67 |         23.4 |          14.9 | 13.8 [12.7, 14.9]         |           13.1 |        2025 |              -1.86 |               -0.67 |
| Tyrion Ingram-Dawkins | DL      |    139 |    4.86 |         21.4 |          13.6 | 13.6 [12.6, 14.7]         |           11.8 |        2025 |              -1.81 |               -1.77 |
| Xavier Legette        | WR      |     32 |    4.39 |         24.9 |          19.5 | 18.3 [17.7, 19.0]         |           17.9 |        2024 |              -1.6  |               -0.48 |
| R.J. Mickens          | DB      |    214 |    4.49 |         24.5 |          17.9 | 17.4 [16.7, 18.1]         |           16.5 |        2025 |              -1.41 |               -0.88 |
| Marques Sigle         | DB      |    160 |    4.37 |         24.4 |          17.8 | 17.3 [16.6, 18.0]         |           16.4 |        2025 |              -1.41 |               -0.94 |
| Byron Murphy          | DL      |     16 |    4.87 |         20.6 |          12.9 | 12.9 [12.0, 13.9]         |           11.6 |        2024 |              -1.33 |               -1.34 |
| Darien Porter         | DB      |     68 |    4.3  |         24.9 |          18.1 | 17.5 [16.8, 18.3]         |           16.9 |        2025 |              -1.28 |               -0.69 |


**Plays faster than he tests** (naive projection far below the game):

| name              | group   | pick   |   forty |   vmax (mph) |   naive (mph) | Sunday proj (mph) [80%]   |   actual (mph) |   draftYear |   miss naive (mph) |   miss shrunk (mph) |
|:------------------|:--------|:-------|--------:|-------------:|--------------:|:--------------------------|---------------:|------------:|-------------------:|--------------------:|
| Keondre Coburn    | DL      | 194    |    5.22 |         18.8 |          11.9 | 12.3 [11.4, 13.3]         |           14   |        2023 |               2.03 |                1.64 |
| Eric Watts        | DL      | UDFA   |    4.67 |         22   |          13.8 | 13.6 [12.6, 14.6]         |           15.6 |        2024 |               1.8  |                2    |
| Marshawn Kneeland | DL      | 56     |    4.75 |         21.4 |          13.4 | 13.6 [12.7, 14.7]         |           15.2 |        2024 |               1.77 |                1.56 |
| Terrion Arnold    | DB      | 24     |    4.5  |         22.6 |          16.2 | 16.7 [16.0, 17.4]         |           17.8 |        2024 |               1.59 |                1.08 |
| McKinnley Jackson | DL      | 97     |    5.26 |         18.7 |          11.7 | 12.1 [11.2, 13.1]         |           13.2 |        2024 |               1.54 |                1.13 |
| Cam Hart          | DB      | 140    |    4.5  |         22.9 |          16.4 | 16.7 [16.0, 17.4]         |           17.9 |        2024 |               1.48 |                1.2  |
| Austin Booker     | DL      | 144    |    4.79 |         21.2 |          13.3 | 14.2 [13.1, 15.4]         |           14.6 |        2024 |               1.35 |                0.46 |
| Javon Solomon     | DL      | 168    |    4.72 |         21.7 |          13.6 | 14.1 [13.1, 15.2]         |           14.9 |        2024 |               1.32 |                0.77 |


## 5. Payoff: illustrative separation value on vertical routes (WR)

**Rough illustration, not a validated effect.** Within a player, a 1-SD faster vertical rep (SD = 1.01 yd/s within player × route) comes with +0.11 yd separation. Across WRs, 1 yd/s of p99 goes with 0.74 yd/s of mean vertical-route peak speed. Chaining those gives about 0.078 yd of separation per yd/s (0.038 yd per mph) of projected Sunday speed, relative to the average WR. This borrows a within-player slope for between-player differences.

| player                           |   vmax (mph) |   naive proj (mph) |   shrunk proj (mph) |   sep. edge naive (yd) |   sep. edge shrunk (yd) |
|:---------------------------------|-------------:|-------------------:|--------------------:|-----------------------:|------------------------:|
| fastest WR vmax: Isaiah Bond     |       25.774 |             20.019 |              19.326 |                  0.065 |                   0.04  |
| slowest WR vmax: Isaiah Williams |       21.693 |             16.849 |              17.917 |                 -0.055 |                  -0.014 |


## 6. Pre-draft view: prospects with a tracked 40 but no usable NFL target (earliest picks)

This shows how the tool is used before the draft: a projection and band from Combine data alone.

| name            | group   |   draftYear |   pick |   REG plays |   forty |   vmax (mph) |   naive (mph) | Sunday proj (mph) [80%]   |   sep. edge, illustrative (yd) |
|:----------------|:--------|------------:|-------:|------------:|--------:|-------------:|--------------:|:--------------------------|-------------------------------:|
| Isaiah Foskey   | DL      |        2023 |     40 |         181 |    4.58 |         22.9 |          14.5 | 13.9 [12.9, 15.0]         |                                |
| Cam Smith       | DB      |        2023 |     51 |         153 |    4.43 |         24.3 |          17.5 | 17.5 [16.7, 18.2]         |                                |
| Kyle Williams   | WR      |        2025 |     69 |         178 |    4.4  |         23.8 |          18.5 | 18.5 [17.9, 19.1]         |                           0.01 |
| Landon Jackson  | DL      |        2025 |     72 |          30 |    4.68 |         22.4 |          14.2 | 13.9 [12.9, 15.0]         |                                |
| Bralen Trice    | DL      |        2024 |     74 |           0 |    4.72 |         21.5 |          13.6 | 14.4 [13.3, 15.6]         |                                |
| Jermaine Burton | WR      |        2024 |     80 |          96 |    4.45 |         23.9 |          18.5 | 18.4 [17.8, 19.0]         |                           0    |
| Roman Wilson    | WR      |        2024 |     84 |         155 |    4.39 |         23.7 |          18.4 | 18.6 [17.9, 19.2]         |                           0.01 |
| Caleb Ransaw    | DB      |        2025 |     88 |           0 |    4.33 |         23.7 |          17.1 | 17.0 [16.3, 17.7]         |                                |
| Elijah Jones    | DB      |        2024 |     90 |          37 |    4.44 |         24.6 |          17.7 | 17.4 [16.7, 18.2]         |                                |
| Jaylin Smith    | DB      |        2025 |     97 |          31 |    4.45 |         25.2 |          18.2 | 17.6 [16.8, 18.4]         |                                |
| Siaki Ika       | DL      |        2023 |     98 |          99 |    5.39 |         18   |          11.4 | 12.0 [11.1, 13.0]         |                                |
| Tai Felton      | WR      |        2025 |    102 |          21 |    4.37 |         25.1 |          19.5 | 19.1 [18.4, 19.7]         |                           0.03 |


All 289 WR/DB/DL prospects with a tracked 40 are in the parquet (154 with an NFL target, 135 without).
