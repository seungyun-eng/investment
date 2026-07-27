# Macro + momentum SP500 methodology

## Research question

The workflow tests two separate predictive hypotheses before testing allocation:

1. Can information available at the close of day \(t\) predict whether SPY will
   suffer a material drawdown over the next \(h\) trading days?
2. Can the same information predict SPY's forward return in excess of cash?

The adjusted SPY series is a learning proxy for the broad US equity market. The
analysis does not assume that an SP500 result transfers to an individual stock.

## Targets

For adjusted close \(P_t\), annual cash rate \(r_t^c\), horizon \(h\), and
drawdown threshold \(d < 0\):

\[
R_{t,h} = \frac{P_{t+h}}{P_t} - 1
\]

\[
R^c_{t,h} =
\exp\left(\sum_{j=1}^{h}\frac{\log(1+r^c_{t+j})}{252}\right)-1
\]

\[
ER_{t,h} = R_{t,h} - R^c_{t,h}
\]

\[
Y^{risk}_{t,h,d} =
\mathbf{1}\left(
\min_{1\le j\le h}\left[\frac{P_{t+j}}{P_t}-1\right] \le d
\right)
\]

Return horizons are 21, 63, 126, and 252 trading days. Risk horizons are 21,
63, and 126 days with -5%, -10%, -15%, and -20% labels. The primary selection
targets are 126-day excess return and a 126-day -10% drawdown event.

## Feature families

- Momentum: 5, 10, 21, 42, 63, 126, 189, and 252-day returns.
- Trend: price/SMA ratios and SMA slopes for 20, 50, 100, and 200 days.
- Risk: realized and downside volatility for 10, 20, 60, and 126 days;
  252-day drawdown and 63-day rebound.
- Volatility regime: VIX level, changes, trailing z-scores and percentiles,
  VIX/VIX3M term structure.
- Rates and macro: 10y-2y and 10y-3m curves, Fed Funds, CPI,
  unemployment, WTI, NFCI, Treasury and corporate yields.
- Credit: official HYOAS where available, legacy high-yield effective yield
  truthfully labelled `HYYield`, and `HYYield - GS10` truthfully labelled as an
  excess-yield proxy.
- Interactions: selected trend × macro and drawdown × credit terms.
- Macro confirmation channels: volatility, credit, NFCI financial conditions,
  labor, and the yield curve. Each channel averages logistic transforms of
  trailing z-scores so neutral conditions map near 0.5. A separate trend
  channel applies the same transform to standardized 21-day deterioration.

For available macro channels \(c\), let \(z^{level}_{c,t}\) and
\(z^{trend}_{c,t}\) use a sign convention where larger means more stress. Then:

\[
L_t = \operatorname{mean}_c \sigma(z^{level}_{c,t}), \qquad
T_t = \operatorname{mean}_c \sigma(z^{trend}_{c,t})
\]

\[
M_t = 0.65L_t + 0.35T_t
\]

Credit pairs high-yield effective yield with BAA-10y, volatility pairs VIX with
VIX/VIX3M term structure, and the yield-curve channel reverses the signs of
10y-3m and 10y-2y so deeper inversion maps to more stress. Missing channel
members are ignored rather than filled from future data.

Trailing distribution statistics exclude the current observation from their
historical benchmark. Monthly data receives a conservative 45-calendar-day
availability lag; NFCI receives seven days.

## Nested walk-forward validation

For outer test year \(y\):

- training observations are limited to years \(y-10\) through \(y-1\);
- any row whose target end date reaches the test period is purged;
- the last three training years form expanding, time-ordered inner validation
  folds;
- model family, feature group, and hyperparameters are selected only on those
  inner folds;
- the selected specification is refit and used to predict year \(y\).

Classifiers compete between logistic regression and histogram gradient
boosting. Regressors compete between ridge regression and histogram gradient
boosting. The default budget creates roughly 2,880 inner-fold candidate
evaluations over the full 2007–2026 run.

The robustness challenger in `research_expanding_weekly.json` changes only the
estimation sample: it uses all history available before each outer year and
keeps every fifth trading observation for training and inner validation. It
still predicts every OOS trading day. This reduces duplicated overlapping
labels and prevents a ten-year window from forgetting older rare crises.
Allocation thresholds remain unchanged, so the challenger is not an
allocation-curve fit.

Classifier selection balances discrimination and calibration:

\[
S_{class} = AUC - Brier
\]

Regression selection balances ordering and absolute error:

\[
S_{reg} = Spearman - 0.25\frac{MAE}{\sigma(y)}
\]

Reported metrics include AUC, Brier score, average precision, calibration,
Spearman correlation, MAE, RMSE, and predicted-quintile spread.

## Original stateless allocation baseline

Let \(p_t\) be the mean predicted drawdown probability and
\(\hat{ER}_t\) the mean selected 63/126-day predicted excess return.

\[
w_t =
\begin{cases}
0.25, & p_t \ge 0.60 \land \hat{ER}_t < 0 \\
0.70, & p_t \ge 0.50 \land \hat{ER}_t < 0 \\
1.00, & \text{otherwise}
\end{cases}
\]

The weight decided at close \(t\) is eligible only at the next session's open.
The backtest applies 5 bps transaction cost and 5 bps slippage per trade,
accrues the unused cash balance, and uses a five-percentage-point rebalance
band. ROI follows the repository definition:

\[
ROI = \left(\frac{FinalValue}{TotalInjected}-1\right)\times100
\]

Sensitivity tables vary risk thresholds, defensive weights, and costs. A
21-day block bootstrap estimates uncertainty in annualized excess return versus
Buy & Hold.

## Stateful macro allocation challenger

The original rule is retained unchanged for comparison. The primary challenger
removes its unstable expected-return gate and uses:

\[
p_t = 0.40\hat p^{63}_t + 0.60\hat p^{126}_t
\]

The risk score receives a 10-day exponential smoothing window and the macro
confirmation score \(M_t\) receives a 20-day exponential smoothing window.
Normal transitions are evaluated only on the final available trading session
of each week. An emergency can move directly to defensive on any trading day.

| Current state | Confirmed condition | Next state | SPY weight |
|---|---|---|---:|
| Normal | risk >= 0.55 and macro >= 0.55 | Caution | 70% |
| Normal/Caution | risk >= 0.70 and macro >= 0.60 | Defensive | 25% |
| Normal | smoothed macro >= 0.60 and raw macro >= smoothed macro | Caution | 70% |
| Normal/Caution | smoothed macro >= 0.70 and raw macro >= smoothed macro | Defensive | 25% |
| Any non-defensive | (risk >= 0.80 and macro >= 0.65) or raw macro >= 0.85 | Defensive immediately | 25% |
| Caution/Defensive | standard relief or confirmed momentum recovery | Recovery | 70% |
| Recovery | standard relief or confirmed momentum recovery | Normal | 100% |

Non-emergency transitions require two consecutive weekly confirmations.
The macro-only route is intentional: the 2020 and 2022 audit showed that the
explicit macro composite reached stressed levels while the fitted drawdown
model stayed low, so requiring model agreement in every case would make macro
data only decorative. Requiring raw stress to remain above its smoothed level
prevents a high but already-falling stress reading from causing a late sale.
Caution/defensive states must remain active for at least 20 trading sessions;
recovery lasts at least 10 sessions; and a return to normal has a 10-session
cooldown before another ordinary risk reduction.

Standard relief is risk <= 0.45 and macro <= 0.50 when leaving caution or
defensive, and risk <= 0.50 and macro <= 0.52 when returning to normal.
Momentum recovery is separately defined as risk <= 0.50, macro <= 0.60, raw
macro below smoothed macro, positive 63-session SPY momentum, and SPY above its
126-session moving average. It permits recovery before slow labor or credit
series have fully normalized, but only after the minimum hold and weekly
confirmation rules.

When the current close is more than 0.25% below the close where the strategy
most recently returned to normal, an ordinary normal-to-caution reduction is
blocked unless risk is at least 0.65 or macro confirmation is at least 0.65.
This is a signal reference, not tax-lot or realized-P&L accounting, and the
0.80/0.65 joint emergency and 0.85 macro-only emergency overrides remain
available. The rule therefore suppresses weak loss sales without promising
never to sell at a loss during a genuine risk escalation.

Stateful rebalancing uses a ten-percentage-point band. Sensitivity analysis calls
this exact state signal function while varying only documented nearby entry
confirmation and minimum-hold parameters. The primary 21-day block bootstrap
compares `StatefulMacro` with Buy & Hold; the stateless bootstrap is saved
separately.

## Interpretation rule

Allocation results are not considered meaningful unless predictive metrics are
stable across outer years, risk probabilities are reasonably calibrated, and
return predictions show a positive, persistent quintile spread. A strategy that
only wins through one allocation threshold is treated as overfit.
