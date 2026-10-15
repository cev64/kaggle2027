# Speed Tax step 3: payoff tests

_Generated 2026-10-15 by `scripts/speed_tax_payoff.py`._

## Summary (interpretation as of 2026-10-15)

**What holds up**
- **Top speed translates almost one for one.** The elasticity of game peak speed on Combine vmax is 0.82
  [0.61, 1.04]. **Burst translates at about half**: 0.42 [0.21, 0.64], or about 0.5 after correcting for
  Combine retest error. The start from a 3-point stance and a route release are different movements.
- **Within a player, faster reps pay on vertical routes**: +0.10 yd separation per within-player SD
  [0.06, 0.15]. On breaking routes, faster reps are targeted more often (+2.8 pp per SD) with no
  separation change. On short routes, faster reps have *less* separation (-0.18 yd per SD). Receivers who
  settle into a hitch or sit in zone are slower and more open, so the direction there is not causal.
- **The tax is a stable trait** (year-over-year r about 0.87). It is unrelated to draft slot.

**What does not hold up**
- **Combine-relative speed does not beat absolute speed.** Separation is explained as well or better by
  absolute peak speed. A given speed is not worth less to a faster player (vmax term near 0 on vertical
  routes). The "share of capacity" framing adds nothing as an explanatory variable.
- **At the player level, nothing tracking-based adds to draft slot, forty and size**, whether for
  separation over expected, target rate over expected, or even in-game speed and burst levels (all
  ΔR² about 0, n = 75). The one positive, burst2_10 predicting the burst tax, is mechanical: the tax
  is defined relative to burst2_10.

**Implication.** The tracked 40 confirms and sharpens what the stopwatch already says about top speed. Its
burst dimension is reliable at the Combine but largely does not carry over to route releases. A writeup
built on "the share of capacity used predicts production" is not supported. See the reply in the session
log for the options this leaves.


## 1. Combine-to-game translation (elasticities)

A mixed model of log(game metric) on log(Combine capacity), with route/window/alignment/coverage/cushion/field/down context and a player random intercept. β = 1 means a receiver with 10% more Combine capacity shows 10% more on Sundays. The intercept is the denominator-corrected tax (`*TaxLog`). Combine burst has retest ICC 0.82, so its β is attenuated by about that factor.


| model                                            | term    |   coef |   lo95 |   hi95 |   p |     n |   players |
|:-------------------------------------------------|:--------|-------:|-------:|-------:|----:|------:|----------:|
| log game peak speed ~ log Combine vmax           | logCapC |  0.822 |  0.605 |  1.038 |   0 | 32087 |        85 |
| log game burst (2-10 yd) ~ log Combine burst2_10 | logCapC |  0.422 |  0.205 |  0.639 |   0 | 16594 |        78 |


## 2a. Within-player payoff: separation and target on speed (player FE, cluster-robust SE)

`speedZ` is the rep's peak speed, standardized within player × route (min 5 reps). The coefficient is the change per within-player SD. Context and player fixed effects are included.


| outcome         | family   |   coef |   lo95 |   hi95 |     p |     n |   players |
|:----------------|:---------|-------:|-------:|-------:|------:|------:|----------:|
| separation (yd) | vertical |  0.101 |  0.055 |  0.147 | 0     | 11575 |        89 |
| separation (yd) | breaking |  0.001 | -0.048 |  0.051 | 0.954 | 10275 |        89 |
| separation (yd) | short    | -0.177 | -0.234 | -0.12  | 0     |  9148 |        87 |
| target (LPM)    | vertical |  0.009 | -0.001 |  0.018 | 0.072 | 11575 |        89 |
| target (LPM)    | breaking |  0.028 |  0.017 |  0.039 | 0     | 10275 |        89 |
| target (LPM)    | short    |  0.009 | -0.002 |  0.02  | 0.091 |  9148 |        87 |


## 2b. Absolute vs Combine-relative speed (mixed models, ML)

Same context and a player random intercept. Positive `logLik - absolute` means Combine-relative speed explains separation better than absolute speed. `vmaxC` tests whether a given absolute speed is worth less for a faster player (negative = speed is judged relative to capacity).


| family   | model                   |   logLik - absolute | key term   |   coef |     p |     n |
|:---------|:------------------------|--------------------:|:-----------|-------:|------:|------:|
| vertical | absolute speed          |               0     | peakC      |  0.11  | 0     | 11575 |
| vertical | Combine-relative speed  |              -1.505 | fracC      |  0.121 | 0     | 11575 |
| vertical | absolute + Combine vmax |               0.049 | vmaxC      |  0.015 | 0.755 | 11575 |
| breaking | absolute speed          |               0     | peakC      |  0.003 | 0.864 | 10275 |
| breaking | Combine-relative speed  |              -0.004 | fracC      | -0.003 | 0.882 | 10275 |
| breaking | absolute + Combine vmax |               1.845 | vmaxC      |  0.117 | 0.055 | 10275 |
| short    | absolute speed          |               0     | peakC      | -0.151 | 0     |  9148 |
| short    | Combine-relative speed  |              -0.611 | fracC      | -0.17  | 0     |  9148 |
| short    | absolute + Combine vmax |               0     | vmaxC      | -0.001 | 0.984 |  9148 |


## 3a. Player level: incremental value beyond draft slot + forty + size (nested CV, routes >= 100)

Outcomes are empirical-Bayes-shrunk per-player residuals from a context-only model. Delta CIs are bootstrapped over players from one out-of-fold split.


| outcome   | added                     |   n |   cvR2 base |   cvR2 +added |   delta |   delta lo95 |   delta hi95 |
|:----------|:--------------------------|----:|------------:|--------------:|--------:|-------------:|-------------:|
| sepOE     | vmax + burst2_10          |  75 |       0.063 |         0.016 |  -0.047 |       -0.146 |        0.009 |
| sepOE     | speedTaxLog               |  76 |       0.05  |         0.018 |  -0.031 |       -0.068 |       -0.017 |
| sepOE     | burstTaxLog               |  75 |       0.063 |         0.046 |  -0.017 |       -0.046 |        0.026 |
| sepOE     | speedTaxLog + burstTaxLog |  75 |       0.063 |         0.011 |  -0.052 |       -0.047 |        0.027 |
| targetOE  | vmax + burst2_10          |  75 |       0.185 |         0.182 |  -0.003 |       -0.061 |        0.113 |
| targetOE  | speedTaxLog               |  76 |       0.169 |         0.165 |  -0.004 |       -0.061 |        0.033 |
| targetOE  | burstTaxLog               |  75 |       0.185 |         0.201 |   0.016 |       -0.063 |        0.114 |
| targetOE  | speedTaxLog + burstTaxLog |  75 |       0.185 |         0.18  |  -0.005 |       -0.103 |        0.111 |


## 3b. Does the Combine predict the game tax? (base: log pick + size; added: forty, vmax, burst2_10)

| outcome     | added                    |   n |   cvR2 base |   cvR2 +added |   delta |   delta lo95 |   delta hi95 |
|:------------|:-------------------------|----:|------------:|--------------:|--------:|-------------:|-------------:|
| speedTaxLog | forty + vmax + burst2_10 |  75 |       0.25  |         0.412 |   0.162 |        0.024 |        0.326 |
| burstTaxLog | forty + vmax + burst2_10 |  75 |       0.371 |         0.528 |   0.157 |        0.034 |        0.31  |


## 3c. Beyond the stopwatch: tracking capacity -> in-game speed (base: log pick + size + official forty)

`gameSpeedLevel` and `gameBurstLevel` are player random intercepts from context-only models of log peak speed and log 2-10 yd burst (absolute, with no Combine term).


| outcome        | added            |   n |   cvR2 base |   cvR2 +added |   delta |   delta lo95 |   delta hi95 |
|:---------------|:-----------------|----:|------------:|--------------:|--------:|-------------:|-------------:|
| gameSpeedLevel | vmax             |  76 |       0.647 |         0.641 |  -0.006 |       -0.016 |        0.022 |
| gameSpeedLevel | burst2_10        |  75 |       0.632 |         0.632 |   0     |       -0.026 |        0.032 |
| gameSpeedLevel | vmax + burst2_10 |  75 |       0.632 |         0.619 |  -0.013 |       -0.064 |        0.009 |
| gameBurstLevel | vmax             |  75 |       0.619 |         0.607 |  -0.012 |       -0.015 |        0.001 |
| gameBurstLevel | burst2_10        |  75 |       0.619 |         0.613 |  -0.006 |       -0.009 |        0.009 |
| gameBurstLevel | vmax + burst2_10 |  75 |       0.619 |         0.602 |  -0.016 |       -0.024 |        0.006 |
| speedTaxLog    | vmax             |  76 |       0.346 |         0.443 |   0.097 |       -0.016 |        0.198 |
| speedTaxLog    | burst2_10        |  75 |       0.333 |         0.338 |   0.005 |       -0.036 |        0.098 |
| speedTaxLog    | vmax + burst2_10 |  75 |       0.333 |         0.412 |   0.079 |       -0.056 |        0.226 |
| burstTaxLog    | vmax             |  75 |       0.356 |         0.422 |   0.065 |       -0.032 |        0.176 |
| burstTaxLog    | burst2_10        |  75 |       0.356 |         0.542 |   0.185 |        0.068 |        0.338 |
| burstTaxLog    | vmax + burst2_10 |  75 |       0.356 |         0.528 |   0.172 |        0.06  |        0.289 |


## 3d. Player-level Spearman correlations

|             |   sepOE |   targetOE |   speedTaxLog |   burstTaxLog |   vmax |   burst2_10 |   forty |   logPick |
|:------------|--------:|-----------:|--------------:|--------------:|-------:|------------:|--------:|----------:|
| sepOE       |    1    |      -0.09 |          0.1  |          0.12 |   0.08 |        0.06 |   -0.11 |      0.15 |
| targetOE    |   -0.09 |       1    |          0.02 |         -0.1  |   0.05 |        0.09 |   -0.13 |     -0.35 |
| speedTaxLog |    0.1  |       0.02 |          1    |          0.48 |  -0.07 |        0.33 |   -0.14 |     -0.02 |
| burstTaxLog |    0.12 |      -0.1  |          0.48 |          1    |   0.55 |        0.02 |   -0.46 |      0.06 |
| vmax        |    0.08 |       0.05 |         -0.07 |          0.55 |   1    |        0.24 |   -0.85 |     -0.13 |
| burst2_10   |    0.06 |       0.09 |          0.33 |          0.02 |   0.24 |        1    |   -0.57 |     -0.15 |
| forty       |   -0.11 |      -0.13 |         -0.14 |         -0.46 |  -0.85 |       -0.57 |    1    |      0.24 |
| logPick     |    0.15 |      -0.35 |         -0.02 |          0.06 |  -0.13 |       -0.15 |    0.24 |      1    |
