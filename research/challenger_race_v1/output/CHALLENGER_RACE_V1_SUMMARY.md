# Challenger Race v1 — A (vol target) vs B (dip buy) vs V7.3+V2

**Decision: `B_MATTERS_MORE_BUT_BASELINE_IS_BROKEN_FIRST`**

Research only. Nothing committed, nothing deployed, no config or frozen artifact
touched. All contenders share one panel: U001 snapshot (31 tickers), frozen
signal-day rule, 2024-01-02..2026-08-12, 10 bps, next-session-open execution,
`ROI = (final / injected - 1) * 100`.

---

## 0. Blocker found before the race: the champion is not what the dashboard says

Replaying frozen V7.3 on the **current working tree** does not reproduce the
published number.

| | Frozen artifact | Aug-17 audit replay | This replay (2026-08-20) |
|---|---:|---:|---:|
| ROI | 213.72% | 213.31% | **190.01%** |
| CAGR | 54.99% | 54.91% | **50.39%** |
| MDD | -31.99% | -32.08% | -31.58% |
| Rebalances | 43 | 42 | 40 |

Diagnosis:

- Price data is **byte-identical** to the Aug-17 cached panel (27,957 rows, max
  relative difference 0.0). Not a price refresh.
- The **selection** differs on **29 of 138 weekly signal dates**. Examples:
  Jun 2025 `SNOW → MSFT`; Feb–Mar 2026 `AVGO → PLTR`; Apr–May 2026 `PLTR → META`.
- The uncommitted working-tree changes to `sec_filings.py` (+416 lines),
  `filing_v7_optimization.py` (+265) and `signals.py` (+90) move the
  filing/fundamental side of the score.

The frozen-signal-day compatibility shim (from `momentum_chase_audit`) is applied
here and is **not** the cause — it accounts for 2 dates, not 23 pp of ROI.

Every number below is internally consistent (all contenders use this same panel),
but the absolute baseline is 190%, not 213%.

## 1. Challenger A on the stock book (2024-2026): loses

| strategy | CAGR | MDD | Sharpe | Calmar | Turnover |
|---|---:|---:|---:|---:|---:|
| V7.3 (no overlay) | 50.40 | -31.58 | 1.46 | 1.60 | 7.9 |
| V7.3 + V2 (original params) | 49.62 | -31.75 | 1.52 | 1.56 | 7.9 |
| V7.3 + V2 (tuned params) | 54.73 | -25.78 | **1.67** | **2.12** | 7.9 |
| A voltgt15 db10 | 27.18 | -19.56 | 1.50 | 1.39 | 4.3 |
| A voltgt20 db10 | 33.63 | -25.62 | 1.40 | 1.31 | 6.3 |
| A voltgt25 db10 | 37.81 | -29.73 | 1.34 | 1.27 | 7.1 |
| A voltgt30 db10 | 39.96 | -31.30 | 1.32 | 1.28 | 7.2 |
| A invvol-only | 41.16 | -29.15 | 1.40 | 1.41 | 7.6 |
| CONTROL universe equal weight | 27.74 | -25.01 | 1.28 | 1.11 | 2.4 |
| CONTROL SPY buy&hold | 22.10 | -18.76 | 1.35 | 1.18 | — |
| CONTROL QQQ buy&hold | 25.20 | -22.88 | 1.18 | 1.10 | — |

**No A variant beats V7.3-alone on Calmar.** Vol targeting reliably cuts
drawdown and costs more return than it saves. Inverse-vol name weighting is
neutral-to-negative and combining it with the exposure dial is worse than either.

The selection engine does clear its controls: 190% vs 89% universe-equal-weight
vs 80% QQQ vs 68% SPY, over the full window.

## 2. But 2024-2026 cannot judge A — there is no bear market in it

SPY 1995-2026, both mechanisms stripped off the stock book, 10 bps charged on
every weight change, plus a **matched-exposure control** (constant weight equal
to each rule's own mean exposure) that isolates *timing* from *just holding less*:

| rule | Mean exp. | CAGR | MDD | Sharpe | Calmar | Sharpe gain vs flat | Calmar gain vs flat |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 (original) | 0.90 | 10.58 | -43.14 | 0.704 | 0.245 | +0.045 | +0.044 |
| V2 (tuned on 2024-26) | 0.91 | 10.35 | -46.58 | 0.690 | 0.222 | +0.031 | +0.021 |
| A voltgt10 | 0.70 | 7.82 | -27.49 | 0.760 | 0.284 | **+0.102** | **+0.089** |
| A voltgt12 | 0.78 | 8.90 | -32.43 | **0.766** | 0.275 | **+0.107** | +0.077 |
| A voltgt15 | 0.85 | 9.61 | -38.89 | 0.736 | 0.247 | +0.077 | +0.047 |
| buy&hold | 1.00 | 11.27 | -55.19 | 0.659 | 0.201 | — | — |
| V2 + voltgt15, min of both | 0.81 | 9.25 | -31.93 | 0.750 | **0.290** | — | — |

**A's timing is worth roughly twice V2's**, at every setting, across 31 years and
four bear markets. Both are real (both beat their flat control), neither raises
CAGR — they buy drawdown reduction.

Regime detail (ROI within window):

| window | buy&hold | V2 original | A voltgt10 |
|---|---:|---:|---:|
| dotcom 2000-02 | -36.9 | -26.2 | **-21.6** |
| GFC 2007-09 | -15.9 | -0.6 | **+0.2** |
| covid 2020 | +17.2 | **+25.7** | +0.6 |
| bear 2022 | -18.7 | -16.2 | **-10.4** |

V2 owns exactly one regime: the **fast V-shaped crash** (2020), where 60-day
realised vol reacts far too slowly. A owns the slow grinding ones. Taking the
**minimum of the two exposures** gives the best Calmar of anything tested (0.290).

### V2's tuned parameters are confirmed overfit

Tuned beats original on its own tuning window (2024-26 SPY ROI 82.9% vs 70.2%)
and loses on every metric over 1995-2026 (Sharpe 0.690 vs 0.704, MDD -46.6 vs
-43.1, timing value +0.031 vs +0.045). Recommend reverting to original params.

## 3. Challenger B: real edge, in exactly the windows V7.3 bleeds

| strategy | CAGR | MDD | Sharpe | Calmar | Turnover | Corr. w/ champion |
|---|---:|---:|---:|---:|---:|---:|
| CHAMPION V7.3+V2 tuned | 54.72 | -25.78 | **1.67** | **2.12** | 7.5 | — |
| CHAMPION V7.3+V2 original | 49.61 | -31.75 | 1.52 | 1.56 | 7.5 | — |
| B4 dip5 + quality + trend, n5, hold4w | 43.77 | -26.31 | 1.38 | 1.66 | **42.7** | 0.69 |
| B2 RSI dip + quality + trend, n5 | 37.87 | -29.55 | 1.27 | 1.28 | 73.2 | 0.66 |
| B1 dip5 + quality + trend, n5 | 16.17 | -33.36 | 0.68 | 0.48 | 75.4 | 0.65 |
| B3 same, n3 | 23.57 | -38.18 | 0.81 | 0.62 | 114.7 | 0.57 |
| B5 dip5, no gates | 15.18 | -34.17 | 0.60 | 0.44 | 98.6 | 0.50 |
| B6 dip5 + trend only | 2.39 | -49.14 | 0.23 | 0.05 | 83.1 | 0.53 |
| BLEND 50/50 champion + B4 | 49.98 | -25.35 | 1.63 | 1.97 | — | — |

Sub-window ROI:

| strategy | 2024 | 2025 | **2026 OOS** | **tariff 2025-02..05** |
|---|---:|---:|---:|---:|
| V7.3 (no overlay) | 76.7 | 64.7 | **-1.4** | **-1.6** |
| CHAMPION V7.3+V2 tuned | 76.7 | 67.3 | +4.5 | -0.0 |
| CHAMPION V7.3+V2 original | 76.7 | 53.6 | +4.3 | -8.3 |
| B4 | 38.2 | 65.3 | **+11.7** | **+6.6** |
| B1 | 27.6 | 8.9 | +7.6 | **+8.3** |
| CONTROL SPY | 25.6 | 18.0 | **+13.7** | -3.3 |
| CONTROL QQQ | 27.0 | 20.4 | **+18.0** | -3.6 |

Findings:

- B loses the full-period race and needs **5.7x the trading** of the champion.
- B wins the 2026 holdout (+11.7% vs +4.5%) and the tariff drawdown (+6.6% vs
  -8.3% for the honest-parameter champion).
- Correlation 0.69 is **too high to be a clean diversifier**; the 50/50 blend
  does not beat the champion (Sharpe 1.63 vs 1.67).
- **The gates do the work, not the dip.** Removing the quality+growth gate drops
  Sharpe 1.38 → 0.60; trend-gate-only is 0.23 with a -49% drawdown. "Buy the dip"
  as a standalone idea is destructive in this universe.
- Caveat: B4 is the best of 6 hand-built variants, and its holdout win is partly
  what selected it. This is multiple testing on one small window.

## 4. The result that outranks the race

In the 2026 holdout (Jan 2 – Aug 12, ~7.5 months) the entire apparatus is behind
an index fund on **both** return and risk:

| | ROI | MDD |
|---|---:|---:|
| QQQ buy&hold | +18.0% | -11.8% |
| SPY buy&hold | +13.7% | -8.9% |
| B4 challenger | +11.7% | -23.2% |
| Champion V7.3+V2 | +4.5% | -23.8% |
| V7.3 alone | -1.4% | -29.6% |

## 5. Priority order

1. **Resolve the baseline drift.** A champion that silently changed cannot be
   raced. Decide whether the new filing code is an intended upgrade, then re-freeze
   or revert, and restate the dashboard's headline number.
2. **Explain 2026 vs SPY** before building anything new.
3. **B as a stress-window sleeve**, pre-registered: fixed variant, fixed gates,
   fixed hold, declared before looking, and costed at realistic slippage — 42x
   annual turnover will not survive 10 bps in practice.
4. **A as a defense upgrade, not a book:** replace V2's binary alert with
   `min(V2 exposure, vol-target exposure)`, and revert V2's tuned parameters to
   the original ones.

## Files

`run_a.py`, `run_b.py`, `run_a_longhistory.py`, `common.py`, `engine_bits.py`;
outputs `a_results.csv`, `a_subwindows.json`, `b_results.csv`, `b_correlation.csv`,
`b_subwindows.json`, `a_longhistory_results.csv`, `a_longhistory_windows.json`.

---

# Part 2 — widened to 2008-2026

## Data ceiling found first

`SEC Filings/*/filings` starts in **2019** and `Financial_Data_real/` is **empty**.
The filing factor (25% of V7.3) and the PIT-reconstructed growth/quality factors
therefore cannot exist before 2019. What runs back to 2008 is the price side of
the same engine — momentum, trend, risk control, V7.3 MA/MACD/OBV — with the
fundamental weights contributing zero. Called **V7-price** below.

**Survivorship warning:** the universe is today's dashboard list run backwards.
Nobody knew in 2008 that NVDA/AVGO/TSLA belonged on it. Absolute returns here are
fiction. Only *differences* between contenders are readable — they share the
identical biased universe.

## The book, 2008-2026 (35 tickers with price files, top 5, weekly, 10 bps)

| | CAGR | MDD | Sharpe | Calmar | GFC 08-09 | 2022 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V7-price top5 | 30.14 | -47.15 | 1.02 | 0.64 | -30.2 | -29.5 | +49.3 |
| CONTROL universe equal weight | 22.79 | -45.94 | 0.99 | 0.50 | **-0.1** | -30.3 | +9.4 |
| CONTROL SPY buy&hold | 11.42 | -51.87 | 0.65 | 0.22 | -19.4 | -18.6 | +13.7 |
| CONTROL QQQ buy&hold | 15.39 | -49.51 | 0.75 | 0.31 | -9.2 | -33.7 | +18.0 |
| B dip sleeve (price-only gates) | 26.81 | **-64.97** | 0.92 | 0.41 | -37.2 | -40.8 | +9.7 |

Honest turnover (dollars traded / average equity / year — the engine's own
`AnnualizedTurnover` divides by *initial* capital and is unreadable over 18
years): V7-price **8.25x**, B dip sleeve **25.87x**.

Reading: selection beats equal weight on return (30.1 vs 22.8 CAGR) and Calmar
(0.64 vs 0.50), but barely on Sharpe (1.02 vs 0.99) — it is mostly a
return-amplifier, not a risk improver. In the GFC the concentration cost 30 pp
against an equal-weight book that finished flat. **B is worse over the long run
on every metric and carries a -65% drawdown.**

## CORRECTION to Part 1: the vol signal must come from the market, not the book

Part 1 concluded from SPY-only tests that vol targeting beats the MA150 alert.
Applied to the actual book that reverses — and the reason is diagnostic.

Matched-exposure control (each rule vs holding its own mean exposure flat),
2008-2026 on the V7-price book:

| rule | Mean exp. | Sharpe | Calmar | Sharpe gain | Calmar gain |
|---|---:|---:|---:|---:|---:|
| no defense | 1.000 | 1.023 | 0.639 | — | — |
| V2 (MA150 on SPY) | 0.918 | 1.067 | 0.668 | +0.044 | +0.036 |
| voltgt25 driven by **BOOK** vol | 0.876 | 1.038 | 0.620 | +0.015 | **-0.007** |
| voltgt30 driven by **BOOK** vol | 0.909 | 1.029 | 0.593 | +0.006 | **-0.038** |
| voltgt12 driven by **SPY** vol | 0.779 | 1.056 | **0.730** | +0.033 | **+0.112** |
| voltgt15 driven by **SPY** vol | 0.842 | 1.057 | 0.705 | +0.034 | +0.081 |
| **min(V2, SPY-vol voltgt15)** | 0.805 | **1.074** | 0.706 | **+0.051** | +0.086 |

A top-5 book's trailing volatility is dominated by single-name noise, not market
regime: when one holding gets jumpy the rule cuts exposure although nothing about
the market changed, so it sells winners for no reason. SPY's volatility is a real
regime signal. **Drive the target off SPY, apply it to the book.**

GFC 2008-09 ROI: no defense -30.2%, V2 -22.8%, SPY-vol voltgt12 **-15.6%**,
min(V2, SPY-vol voltgt15) -17.5%.

## CORRECTION to Part 1: B's 2026 win was against a broken champion

Over 2008-2026 the dip sleeve is beaten on every metric. Its 2026 win in Part 1
was scored against a V7.3 that itself returned -1.4% that year — not evidence
that B is good.

## The fundamental overlay is both the alpha and the 2026 failure

Controlled ablation, identical 31-ticker U001 snapshot, identical window and
exits, only the growth/quality/filing factors blanked:

| | ROI | CAGR | Sharpe | 2024 | 2025 | **2026 OOS** |
|---|---:|---:|---:|---:|---:|---:|
| V7.3 with filings + fundamentals | 190.0 | 50.4 | 1.46 | +76.7 | **+64.7** | **-1.4** |
| same engine, fundamentals blanked | 75.7 | 24.1 | 0.86 | +63.6 | **-3.4** | **+12.4** |

The fundamental side *is* the strategy — it produced all of 2025 — and it is
exactly what inverted in 2026. (Not a byte-perfect isolation: blanking also
removes the filing coverage/veto gates, and the exit-rank parameters differ
slightly.)

## Revised priority order

1. **Baseline drift** (Part 1 §0) — unchanged, still first.
2. **Why the fundamental side inverted in 2026.** That is where the alpha lives
   and where the failure is; it is one component, not the whole model.
3. **Defense upgrade with a real 18-year edge:** `min(V2, SPY-vol-driven vol
   target)`. Calmar 0.706 vs 0.639 undefended, and it is the only variant that
   beats its matched-exposure control on both Sharpe and Calmar.
4. **B: drop it,** or keep it only as a small tactical sleeve. It loses over
   18 years, draws down 65%, and trades 26x a year.

## Files (Part 2)

`run_long.py`, `run_long_defense.py`, `run_long_addendum.py`; outputs
`long_results.csv`, `long_windows.json`, `long_correlation.csv`,
`long_defense_results.csv`, `long_addendum_results.csv`,
`filing_overlay_ablation.csv`.

---

# Part 3 — 2026: wrong stocks, or right stocks at the wrong time?

**Decision: `TIMING_FAILURE_CONFIRMED_BUT_NOT_TIMEABLE`**

## 1. It is a timing failure, and the size is measurable

Same picks, three clocks, 2026-01-02 .. 2026-08-12:

| | ROI | MDD | Sharpe |
|---|---:|---:|---:|
| **actual** (model's own entries and exits) | **-3.0%** | -29.6 | -0.04 |
| hold_from_first (buy when first picked, never sell) | +4.4% | -18.8 | 0.43 |
| hold_all (same names, bought day 1, never sold) | +11.1% | -21.8 | 0.81 |

Split: of the 14 points lost, ~6.8 went to **entering late** (11.1 → 4.4) and
~7.4 to **selling** (4.4 → -3.0).

`hold_all` is a hindsight bound — you do not know in January which names the
year will pick. `hold_from_first` is achievable: it is just "never sell".

The same decomposition for 2025, as the control:

| | ROI |
|---|---:|
| actual | **+58.1%** |
| hold_all | +43.9% |
| hold_from_first | +30.5% |

**The identical clock added 14 points in 2025 and cost 14 points in 2026.**

## 2. The rule is rotation speed

Sweeping the exit-rank buffer (how far a holding must fall before it is
replaced), selection untouched:

| variant | full CAGR | full Sharpe | turnover | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|
| buffer 1 — rotate fastest | 47.6 | 1.35 | 10.9 | **+82.5** | +49.6 | **-5.8** |
| buffer 4 — as shipped | 50.4 | 1.46 | 7.5 | +76.7 | +58.1 | -3.0 |
| buffer 8 — rotate slowest | 44.5 | 1.38 | 3.6 | **+56.8** | +49.6 | **+5.8** |
| all delays off + buffer 1 | 38.0 | 1.20 | 11.4 | +83.5 | +35.5 | **-12.3** |

Monotone and inverted between regimes: **the faster it rotated, the more it made
in 2024 and the more it lost in 2026.** No fixed speed is right for both.

Code gotcha worth recording: `generate_filing_v7_targets` calls
`policy.strategy_params(base_params)`, which **overrides** `top_k`, `exit_rank`,
`hard_stop_return`, `minimum_hold_rebalances` and `replacement_score_advantage`
from the `FilingV7Policy`. Varying those four on `StrategyParams` is silently
a no-op — a first pass of this ablation produced four identical rows because of it.

## 3. Per-holding evidence, 2026

| Ticker | first held | weeks | run-up 63d before entry | while held | after last held |
|---|---|---:|---:|---:|---:|
| PANW | 2026-06-05 | 11 | **+64.8%** | **+42.3%** | — |
| AVGO | 2026-03-27 | 18 | **-14.1%** | **+38.4%** | — |
| NVDA | 2026-01-02 | 33 | -0.0% | +18.7% | — |
| GOOG | 2026-01-02 | 25 | +28.0% | +16.5% | -6.8% |
| ARRY | 2026-01-02 | 8 | +10.5% | +16.0% | **-53.8%** |
| AAPL | 2026-02-06 | 27 | +3.0% | +12.7% | -3.5% |
| PLTR | 2026-01-02 | 11 | -10.3% | +1.9% | — |
| MSFT | 2026-01-02 | 5 | -8.3% | -9.0% | **+14.4%** |
| META | 2026-02-27 | **23** | +1.9% | **-10.7%** | — |
| NVO | 2026-01-30 | 4 | +15.7% | -20.2% | -2.2% |

- **Momentum chasing was not the problem.** The two entries after the biggest
  run-ups (PANW +64.8%, GOOG +28.0%) were the best and fourth-best trades.
- The two biggest winners only entered in **late March and June** — the book
  spent the first quarter in names that went nowhere.
- Selling ARRY was correct (-53.8% afterwards). Selling MSFT was wrong (+14.4%).
- META was held 23 weeks at -10.7% and was still held at the window end.

## 4. The factor split behind it

Mean weekly cross-sectional rank IC, by horizon and year:

| factor | 3m 2024 | 3m 2025 | **3m 2026** | 6m 2024 | 6m 2025 | **6m 2026** |
|---|---:|---:|---:|---:|---:|---:|
| GrowthFactor | +0.080 | -0.081 | **-0.184** | +0.174 | -0.189 | **-0.138** |
| QualityFactor | +0.041 | -0.059 | **+0.133** | +0.087 | -0.074 | **+0.113** |
| FilingFundamentalFactor | +0.207 | -0.057 | -0.001 | +0.240 | -0.131 | -0.025 |
| TrendFactor | +0.088 | -0.002 | -0.082 | +0.138 | -0.003 | -0.100 |
| AlphaScore | +0.137 | -0.061 | -0.005 | +0.217 | -0.107 | -0.035 |

"The fundamentals broke" is more precisely **growth broke and quality held**.
Growth carries a 0.196 weight and quality 0.309 in V7.3.

## 5. Can the rotation speed be timed? Four detectors say no

Rule tested: freeze rotation (do not trade that week, let the book drift) when a
point-in-time flag fires. Each detector is scored against **500 shuffles that
freeze the same number of weeks at random dates** — a detector only counts if it
sits in the TOP tail of its own shuffle.

| detector | frozen weeks | full Sharpe | **Sharpe percentile vs shuffle** | 2026 ROI |
|---|---:|---:|---:|---:|
| baseline (never freeze) | 0 | 1.46 | — | -3.0 |
| trailing 13w IC < 0 | 16 | 0.95 | **1.8** | -8.8 |
| SPY below 200-day MA | 4 | 1.44 | **32.6** | -4.0 |
| SPY 60d vol above trailing 2y 75th pct | 7 | 1.40 | **30.0** | 0.0 |
| cross-sectional dispersion below trailing median | 18 | 1.23 | **36.6** | +0.8 |

**Every detector lands below the median of freezing at random.** The trailing-IC
one is actively anti-predictive (1.8th percentile). Two of them look acceptable
on 2026 alone (82.6th and 74.6th percentile there) but that is one 7-month
window chosen after the fact, and both are below random over the full period.

Why trailing IC fails is visible in its own inputs: weekly IC has a standard
deviation of 0.21–0.31, so a 13-week average carries a standard error of
0.06–0.09 — larger than the entire year-over-year shift being detected
(2024 +0.054 → 2025 +0.011 → 2026 +0.002). Rank persistence is flat at
0.93–0.94 across all three years and carries no signal at all.

## 6. What this means

- The 2026 picks were fine. **The model's rotation speed was fitted to a
  2024-style market where leadership persisted, and 2026 does not persist.**
- Rotation speed is a real lever with a real trade-off (2024 +82.5/-5.8 at one
  end, +56.8/+5.8 at the other) — but it is a **static** choice, not a timeable
  one. Four detectors, all shuffle-controlled, all failed.
- The more promising lever found here is the **growth/quality split**, not the
  clock: quality kept working in 2026 (+0.133 IC at 3m) while growth inverted
  (-0.184). That is a factor-weight question with a specific, testable direction.

## Files (Part 3)

`run_2026_diagnosis.py`, `run_exit_clock.py`, `run_rotation_regime.py`,
`run_rotation_detectors.py`; outputs `diag_factor_ic_by_year.csv`,
`diag_selection_vs_timing_2026.csv`, `diag_selection_vs_timing_2025.csv`,
`diag_entry_timing_2026.csv`, `diag_exit_clock.csv`, `diag_weekly_ic.csv`,
`diag_rotation_detectors.csv`, `diag_rotation_detectors_control.json`.

---

# Part 4 — both open levers, on longer panels

**Decision: `BOTH_LEVERS_REJECTED`**

Panels: rotation speed on the price-only engine **2008-2026 (19 years)** and on
the full engine **2020-2026 (6.6 years, rebuilt because SEC filing features start
2019-01-09)**; growth/quality on the full engine 2020-2026.

## Q1 — rotation speed: no general edge, and the 2026 result does not generalise

Price-only engine, 2008-2026, exit-rank buffer swept:

| buffer | CAGR | Sharpe | Calmar |
|---:|---:|---:|---:|
| 1 (fastest) | 29.74 | 1.03 | 0.60 |
| 2 | 30.16 | 1.03 | 0.61 |
| **4 (as shipped)** | 30.14 | 1.02 | **0.64** |
| 6 | 28.72 | 1.00 | 0.56 |
| 8 | 27.00 | 0.95 | 0.55 |
| 12 (slowest) | 25.92 | 0.93 | 0.54 |

Year by year, comparing buffer 1 against buffer 8: **slow wins 11 years, fast
wins 8** — a coin flip. But fast's wins are much larger (2021 +89.5 vs -8.2;
2023 +41.2 vs +1.6; 2024 +124.7 vs +84.1), which is why it compounds ahead.
There is no stable regime pattern to trade against.

Full engine, 2020-2026, as the cross-check:

| buffer | CAGR | Sharpe | MDD | turnover | 2022 | 2026 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.17 | 1.07 | -40.19 | 14.3 | -26.3 | -5.8 |
| **4 (shipped)** | 32.93 | **1.12** | -41.87 | 9.8 | -28.7 | -3.0 |
| 6 | 29.23 | 1.05 | -42.30 | 5.2 | -29.0 | +5.4 |
| 8 | 25.57 | 0.99 | **-31.77** | 4.6 | **-15.2** | **+5.8** |
| 12 | 22.08 | 0.90 | -31.33 | 3.7 | -18.6 | -0.1 |

The Part-3 finding reproduces on the longer panel — buffer 8 is better in **both**
bad years (2022 -15.2 vs -28.7, 2026 +5.8 vs -3.0) and cuts MDD by 10 pp — but it
costs 7 pp of CAGR and the shipped buffer still has the best Sharpe.

**Verdict: slowing rotation is a drawdown tool, not a return tool, and Part 3
already established it cannot be switched on and off. Leave the buffer at 4.**

## Q2 — growth vs quality: the Part-3 hypothesis does not survive a portfolio test

Full engine 2020-2026, shifting the growth/quality pool (total 0.505 held constant):

| growth share | CAGR | Sharpe | Calmar | 2022 | 2023 | 2024 | 2025 | 2026 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 all growth | 28.90 | 0.96 | 0.71 | -31.8 | 24.6 | 65.5 | 48.3 | **+20.4** |
| 0.75 | 31.90 | 1.05 | 0.76 | -29.0 | 33.8 | 87.7 | 46.3 | +12.7 |
| 0.50 | 28.78 | 1.00 | 0.71 | -24.7 | 42.0 | 94.1 | 39.7 | +0.6 |
| **0.387 shipped** | 32.94 | 1.11 | 0.79 | -28.7 | 65.3 | 77.1 | 60.0 | -2.9 |
| 0.25 | **33.31** | **1.13** | **0.82** | -27.5 | 70.7 | **103.4** | 51.5 | -1.8 |
| 0.00 all quality | 30.48 | 1.11 | 0.77 | -28.8 | **82.4** | 81.3 | 33.6 | +6.2 |

**Part 3's inference was wrong.** The IC table said quality worked in 2026
(+0.133 at 3m) and growth inverted (-0.184), implying a tilt toward quality.
The portfolio says the reverse: all-growth produced the best 2026 (+20.4%),
all-quality +6.2%, and the shipped mix worst at -2.9%. The relationship is also
non-monotone, which is the signature of noise.

Attribution confirms it is not a factor effect. In 2026 the all-growth variant
traded 8 names and its edge is largely one holding the shipped version never
owned:

| variant | 2026 ROI | names | top 3 contributors |
|---|---:|---:|---|
| all growth | +20.4% | 8 | PANW +7,883 · **SNOW +6,475** · NVDA +3,321 |
| shipped | -2.9% | 10 | PANW +7,095 · AVGO +5,641 · NVDA +2,721 |
| all quality | +6.2% | 8 | NVDA +3,325 · AAPL +2,828 · PANW +2,812 |

SNOW alone is ~a third of the all-growth result over a 7-month window with 8
names. That is a single-name outcome, not evidence about growth weighting.

Over the full 6.6 years the whole sweep spans CAGR 28.8–33.3 and Sharpe
0.96–1.13, with the shipped 0.387 already at 1.11. The best cell (0.25, Sharpe
1.13) beats it in 5 of 7 years by small margins — inside noise for a 5-holding
book.

**Verdict: no change to the growth/quality weights.**

### Method note worth keeping

Cross-sectional rank IC measured over the whole 31-name universe **did not
predict** what a top-5 portfolio would do — it pointed the opposite way in 2026.
IC describes the middle of the cross-section; the strategy only ever owns the
tail. Any future factor decision in this repo should be tested on the portfolio,
not on IC.

## Where this leaves the session

Everything tested across Parts 1–4 was rejected except one item:

| item | verdict |
|---|---|
| A — vol target on the book's own volatility | rejected (worse than its matched-exposure control) |
| B — dip-buy sleeve | rejected (loses over 18 years, -65% MDD, 26x turnover) |
| rotation-speed timing (4 detectors) | rejected (all below random in shuffle control) |
| rotation-speed static change | rejected (drawdown tool only; shipped buffer best Sharpe) |
| growth/quality re-weighting | rejected (contradicts IC, driven by one name) |
| V2 tuned parameters | **revert to original** (overfit to 2024-26) |
| **`min(V2, SPY-vol-driven vol target)`** | **the one survivor** — Calmar 0.706 vs 0.639, beats its matched-exposure control on both Sharpe and Calmar over 2008-2026 |

Still first in line, unchanged since Part 1: **the baseline drift.** The champion
replays at 190% instead of the published 213.7%, and 29 of 138 weeks pick
different stocks, because of uncommitted filing-code changes in the working tree.

## Files (Part 4)

`run_weights_and_speed.py`; outputs `q1_rotation_speed_2008_2026.csv`,
`q1_rotation_speed_2020_2026.csv`, `q2_growth_quality_2020_2026.csv`,
`panel_2020_2026.pkl`.

---

# Part 5 — repair, re-freeze, and the survivor's validation

## Step 1 — the in-flight filing fix is finished

Three tests failed with `KeyError: 'DilutedShares'`, and the failure was in
production code, not stale assertions: the new basic-shares fallback path (for
filers whose diluted counts are split across share classes) left three code
paths reading columns unconditionally. Fixed by degrading gracefully, matching
the convention `add_derived_filing_metrics` already used for
`SharesOutstandingConcept`:

- `DilutedShares` → per-row fallback to `SharesOutstandingSplitAdjusted`
- `_ttm_with_q4_derivation` / `_most_recent_annual` → all-NaN without the
  period schema (every caller is in the display-only valuation block)
- the display-only block's 13 filed-fact inputs → materialised as NaN if absent

`pytest tests/` = **513 passed** (excluding `tests/tesla_v3/`, which is untracked
and was already broken before this session: it imports `wilder_rsi`, which does
not exist in the modified `tsla_integrated/features.py`).

**The repair does not move the baseline.** Repaired = 190.01%, byte-identical to
the broken tree, because the crash paths never trigger for these 31 tickers.
Every number in Parts 1–4 was therefore already computed on repaired
fundamentals — the 2026 findings are not an artifact of the bug.

## Step 2 — what actually caused 213.72% → 190.01%

Not the crash paths. The other two uncommitted fixes:

1. **Q4 derivation in TTM sums.** Most US filers never file a standalone Q4
   10-Q; Q4 exists only inside the annual 10-K. A rolling 4-quarter sum over
   quarterly rows silently skipped every Q4 and wrapped into the next year's Q1.
2. **Split-adjusted share counts.** A 10-for-1 split read as ~1000% dilution.
   Verified on the two names that split inside the window: NVDA's filed share
   growth now reads -0.4% across its June-2024 10-for-1 (correct — a small
   buyback) and AVGO +11-12% (correct — VMware share issuance), not ~+900%.

Drift attribution by ticker (total -23,706):

| Ticker | V7.3 P&L | V7.3.1 P&L | delta |
|---|---:|---:|---:|
| PLTR | 76,157 | 61,795 | **-14,362** |
| ILMN | 7,814 | **0** | -7,814 (no longer selected at all) |
| META | 13,418 | 8,854 | -4,564 |
| MSFT | -11,036 | -7,330 | +3,706 |
| AAPL | 7,571 | 4,351 | -3,220 |
| AVGO | 49,796 | 52,110 | +2,314 |

It is a selection redistribution, not a single arithmetic change.

**Frozen as V7.3.1**, not by overwriting V7.3 — `scripts/dashboard/freeze_backtest.py`
explicitly refuses to re-freeze an existing version, and the registry directory
is untracked so an overwrite would be unrecoverable. `v7_3.json` is kept as the
historical record; a byte-identical backup sits at `v7_3.BACKUP-20260820.json`
(sha256 `9a754a9b…`, matching the value documented in `momentum_chase_audit`).

| | V7.3 | **V7.3.1** |
|---|---:|---:|
| ROI | 213.72% | **190.01%** |
| CAGR | 54.99% | **50.39%** |
| MDD | -31.99% | **-31.58%** |
| Sharpe | 1.535 | **1.464** |

## Step 3 — published numbers corrected

`alpha-desk-cloud/ALPHA_DESK_CLAUDE_HANDOFF.md` (the `213.96% / 55.39%` headline
is withdrawn with the reason), `docs/dashboard_monthly_weight_reset.md` (the
policy decision rests on execution-count reduction, which the repair does not
touch — noted rather than rewritten), `docs/macro/SPY_TACTICAL_DEFENSE.md`
(absolute baseline annotated; its relative comparisons stand), and
`config/dashboard_model_registry/registry.json` (`active_model_version` →
V7.3.1, with `model_version_history`).

## Step 4 — the survivor is REGIME-DEPENDENT, and fails on the live book

Split-sample and parameter sweep, each cell against its own matched-exposure
control (`Sharpe gain` / `Calmar gain`):

| rule | 2008-2017H1 | 2017H2-2026 |
|---|---:|---:|
| **V2 alone** | **-0.064 / -0.041** | **+0.139 / +0.119** |
| SPY-vol target 12% alone | **+0.159 / +0.294** | -0.072 / +0.030 |
| min(V2, SPY-vol 12%) | +0.076 / +0.136 | +0.008 / +0.054 |
| min(V2, SPY-vol 15%) | +0.011 / +0.033 | +0.078 / +0.140 |
| min(V2, SPY-vol 18%) | **-0.010** / +0.003 | +0.087 / +0.103 |
| min(V2, SPY-vol 22%) | **-0.035 / -0.007** | +0.109 / +0.104 |

Two findings that matter more than the headline:

- **V2 alone loses in the first half.** The MA150 alert is worse than holding
  less over 2008-2017H1. Its entire documented value comes from the second half.
- **The two mechanisms are complementary by era**, which is exactly why the
  minimum of them clears both halves — but only at target vols **10–15%**.
  At 18% and 22% it fails the first half. That is genuine parameter sensitivity.

**And on the live V7.3.1 book (2024-2026) every vol-target variant is negative
against its flat control** (Sharpe gains -0.03 to -0.33); only V2 alone is
positive there (+0.053 Sharpe, -0.023 Calmar). That window has no bear market,
which is the same caveat that applies to everything measured on it — but it
cannot be waved away either.

**Verdict: `SHADOW_ONLY_AT_12_TO_15_PCT`.** It survives a split-sample
control on the 19-year price-only book at 10-15% target vol, and it fails on the
recent fundamental book. That is weaker than Part 2 implied, and Part 2's
statement that it "beats its matched-exposure control on both Sharpe and Calmar"
is true only for the full period, not for every sub-window.

## Files (Part 5)

`freeze_v7_3_1.py`, `run_survivor_validation.py`; outputs
`survivor_validation.csv`, `survivor_validation_live_book.csv`,
`repaired_attribution.pkl`. Registry artifact:
`config/dashboard_model_registry/frozen_backtests/v7_3_1.json`.

---

# ERRATUM — a one-day lag error invalidates several conclusions in Parts 1–4

Found 2026-08-20 while checking a different number.

## The bug

`v2_weights` already lags: it decides at index `i` using only day `i-1` data.
Every overlay helper in this study then applied `weights.shift(1)`, delaying both
the de-risk and the re-entry by an **extra** session. Affected:
`engine_bits.v2_overlay_nav`, `run_long_defense.apply_exposure`,
`run_a_longhistory.apply_weights` — and therefore Parts 1, 2, 4 and 5.

It was caught because `artifacts/v7_v2_overlay/tuning_results.json` reproduces
its `spy_buy_hold` reference exactly (163.41%) but not its `standalone_spy`
figure — isolating the discrepancy to the weight application, not the data.

The extra lag penalises V2 far more than a vol target, because V2 switches
abruptly and often (12 episodes in 2022) while a vol target drifts.

## The convention question underneath it

`v2_weights` records its trade price as `close[i]`, so three readings exist and
they are far apart. Settled with **real open prices** (SPY 1995-2026, dashboard
benchmark file, unadjusted, 10 bps):

| rule | ROI | CAGR | MDD | Sharpe |
|---|---:|---:|---:|---:|
| V2 original — A: executed at close[i] (what Parts 1-4 used) | 1572.1 | 9.32 | -46.31 | 0.631 |
| V2 original — B: close-to-close, no lag (what the shipped tuning used) | 2800.5 | 11.24 | -44.93 | 0.740 |
| **V2 original — REAL: signal at close[i-1], executed at open[i]** | **2153.4** | **10.36** | **-46.27** | **0.691** |
| V2 tuned — A | 1393.6 | 8.93 | -54.23 | 0.612 |
| V2 tuned — B | 3363.8 | 11.87 | -42.73 | 0.778 |
| **V2 tuned — REAL** | **2397.1** | **10.72** | **-44.13** | **0.714** |
| buy&hold | 1587.4 | 9.35 | -56.47 | 0.566 |

**A understates V2, B overstates it, and the truth is in between.** The gap
between the two wrong conventions is larger than every effect this study
was trying to measure.

## What is retracted

1. **"V2's tuned parameters are confirmed overfit — revert to the original ones"
   (Parts 2 and 5). RETRACTED.** Under correct open-price execution tuned beats
   original over 31 years too (Sharpe 0.714 vs 0.691, MDD -44.1% vs -46.3%). The
   earlier verdict came from the double lag, which punishes the whipsaw-prone
   tuned rule hardest. The shipped parameters survive; do not revert them.
2. **"A's timing is worth roughly twice V2's" (Part 2). RETRACTED.** With the lag
   removed, V2 beats every vol-target setting on SPY (Sharpe gain +0.180 vs
   +0.109 at the best vol target).
3. **"Neither mechanism raises CAGR — they buy drawdown reduction" (Part 2).
   RETRACTED.** V2 raises CAGR *and* cuts drawdown (10.36% / -46.3% vs
   buy&hold 9.35% / -56.5%).
4. **"min(V2, SPY-vol target) is the one survivor" (Parts 2 and 5). RETRACTED.**
   Plain V2 is the better defense once the lag is fixed.

## What still stands

- The baseline-drift diagnosis and the V7.3.1 re-freeze (Part 5 steps 1-3) — no
  V2 overlay is involved in any of it.
- The 2026 timing decomposition (Part 3): actual -3.0% vs +11.1% held passively.
  No overlay involved.
- Challenger B's rejection (Parts 1, 2) — the dip sleeve carries no V2 overlay.
- The rotation-speed and growth/quality sweeps (Part 4) — no overlay.
- The retraction of the old **236.8%** swap-into-SPY figure: swap still loses
  badly to de-risk under every convention.

## Corrected book-level numbers are a BAND, not a point

The book-level V7.3+V2 runs exist only under conventions A and B, and the
portfolio NAV has no open-price series to redo them properly:

| window | A (too low) | B (too high) |
|---|---:|---:|
| V7.3 + V2 original, 2024-2026 | 186.1% | to be recomputed |
| V7.3 + V2 original, 2020-2026 | 594.6% | 746.2% |
| 2022 alone | -32.9% | -17.7% |

Pinning these down needs the portfolio simulated with the overlay *inside* the
execution engine (which already trades at next open) rather than applied to a
finished NAV curve. Until then, quote the band.
