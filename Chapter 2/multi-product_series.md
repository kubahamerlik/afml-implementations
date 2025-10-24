# Dealing with Multi-Product Series (Advances in Financial Machine Learning, Ch. 2.4)

## ETF Trick – What Is It?

- **Problem:** Strategies trading a spread of futures need to handle things like contract rolls and spread weights properly. Naively combining multiple contracts or using raw prices can create misleading series for modeling and trading.

- **Goal:** 
    - Create a synthetic time series that tracks the value of $1 invested in a spread (basket) of instruments, as if it was an ETF.
    - Handles all nuisances like negative prices, roll dates, and non-constant weights, producing a smooth, model-friendly PnL/price series.

### Key Steps:

1. **Data Columns You Need:**
   - Open/close prices for each instrument at each time
   - Value of one "point" in USD (includes FX if needed)
   - Volume per instrument per bar
   - Carry/dividend/coupon per instrument per bar
   - All instruments must be tradeable at every rebalancing/roll

2. **Holdings Representation:**
   - Portfolio value is tracked by recursively updating value for each time point depending on current holdings, weights, and dividends.
   - At rebalance (including rolls), the allocations/holdings are updated to new weights.
   - Costs (rebalance, bid-ask, volume/tradability) should be included for realism.

3. **Why Do It?**
   - Simulates trading a portfolio as if it was an ETF, not just switching contracts or naïvely chaining historical prices.
   - Avoids negative values, artificial jumps, and problems due to contract roll or weight-change events.

### Implementation Hints

- Treat rebalancing/rolls as "portfolio events" with explicit logic for updating allocations/holdings.
- Model physical costs (bid-ask, rebalancing, volume constraints) to avoid fictitious PnL.
- Use vectorized calculations to efficiently simulate long time series.

---

## PCA Weights for Multi-Product Series

- **Goal:** Allocate portfolio risk optimally across multiple instruments/futures.
- **What is PCA?** Principal Component Analysis finds uncorrelated axes (principal components) capturing the main drivers of variation/risk in your multi-asset set.
- **Application:**
    - Use PCA to allocate risk by projecting portfolio risk onto key principal components.
    - Can define a custom risk distribution so only principal components with the lowest (or highest) variance get allocation.
    - Methods from code snippet: use eigenvalues/eigenvectors from covariance matrix, re-scale to desired risk allocation.

- **Benefits:** 
    - Eliminates noise/overfitting that comes from allocating risk to all assets equally or naively.
    - Lets you focus only on the factors that truly drive risk in the spread/portfolio.

---

## Single Future Roll

- **Problem:** For a single future you need to "chain" contracts as they expire. Rolling incorrectly creates artificial jumps, which mislead models.
- **Solution:** Compute/track cumulative "roll gaps" (differences in price between expiring and new contract at each roll) and either:
    - Subtract backward: Force continuity at the **end** of the raw series.
    - Add forward: Force continuity at the **start** of the raw series.
    - Either approach lets you build a continuous series (do not model the fictitious profit/loss caused by a naive roll).

- **Python Hints:**
    - Keep a running total of roll gaps and apply to the price series.
    - Calculate return series on the continuous (rolled) prices for clean, non-negative investment tracking.

---

## Key Takeaways

- **ETF Trick:** Build a synthetic series for multi-asset portfolios that handles all the real-life nuisances of futures trading like rolls, weights, and transaction costs.
- **PCA Weights:** Use principal components to allocate risk, reducing noise and focusing on “real” drivers of risk.
- **Rolling Futures:** Handle contract expiry properly by adjusting for spread/gaps, not just stitching contracts, to get a realistic, continuous series for analysis.

# 2.5 Sampling Features - Notes

## The Core Problem

When you're working with financial machine learning, you have a continuous stream of bars (whether tick, volume, or dollar bars). But here's the issue - ML algorithms don't work well when you throw ALL the bars at them. There's simply too much data, and most importantly, most bars are just noise. They don't contain useful information.

So the key question becomes: Which observations should we actually sample for our feature matrix?

## Section 2.5.1: Sampling for Reduction

The goal here is simple - reduce the number of observations fed to your ML algorithm, but do it intelligently.

**Two Common (But Bad) Approaches:**

**First approach - Linspace sampling:** This is where you just take every Nth bar. For example, you might take every 100th bar. The problem? The step size is completely arbitrary. You might miss important events. Plus, you're oversampling during quiet market periods and undersampling during volatile periods when interesting things are happening.

**Second approach - Uniform random sampling:** Here you just randomly select, say, 10% of bars. The problem with this is that your results depend on the random seed. You're not focusing on informative observations - you might sample noise and miss the actual signals.

**Why both fail:** Neither method identifies the subset of most relevant observations in terms of predictive power or informational content. The step size is arbitrary, and uniform sampling just doesn't target what matters.

## Section 2.5.2: Event-Based Sampling

This is where things get interesting. The core principle is: sample only when something significant happens.

**What constitutes an "event"?**

Events can be many things:

- A structural break (a regime change in market dynamics)
- An extracted signal from another model or indicator
- Microstructural phenomena (like unusual order flow or toxicity)
- Macroeconomic releases (like NFP, CPI, Fed announcements)
- A volatility spike
- A spread departing from its equilibrium level

**The research process:**

First, you define what constitutes a significant event. Then you sample bars when those events occur. You train your ML algorithm to predict outcomes given the event-triggered features. You evaluate whether predictions are accurate. And finally, you iterate - if there's no predictive power, you redefine what events are or try different features.

The key advantage here is that you're building a quality-filtered dataset where each observation is potentially informative, not just random noise.

## Section 2.5.2.1: The CUSUM Filter

This is the practical implementation of event-based sampling that López de Prado recommends.

**What is CUSUM?**

CUSUM stands for Cumulative Sum. It's a quality-control method designed to detect a shift in the mean value of a measured quantity away from a target value. It comes from industrial statistics but works beautifully for financial data.

**How it works mathematically:**

You start with S₀ = 0. At each bar t, you update: S_t = max{0, S_{t-1} + y_t - E_{t-1}[y_t]}, where y_t is your returns (or other variable) and E_{t-1}[y_t] is the expected value (often 0 for returns).

You trigger an event when S_t ≥ h, where h is your threshold (the filter size). After triggering, you reset S_t = 0.

**Intuitive understanding:**

In a normal market with no trend, returns oscillate around zero: +0.1%, -0.2%, +0.05%, -0.1%. S_t stays near 0, so no events are triggered.

But when a trend emerges, you get consecutive positive returns: +0.3%, +0.2%, +0.4%, +0.3%. Now S_t accumulates: 0.3 → 0.5 → 0.9 → 1.2. When S_t ≥ h, you trigger an EVENT! You reset S_t = 0 and sample this bar.

**Symmetric CUSUM:**

To detect both upward and downward trends, you use a symmetric version. You maintain two accumulators: S_t^+ for upward movements (using max{0, ...}) and S_t^- for downward movements (using min{0, ...}). You trigger when either exceeds the threshold h.

**Python implementation from the book (Snippet 2.4):**

The function getTEvents takes in your raw series (gRaw) and threshold (h). It maintains two accumulators: sPos for upward CUSUM and sNeg for downward CUSUM. For each bar, it updates both accumulators. When sNeg drops below -h, it resets to 0 and adds that timestamp to the event list. When sPos rises above h, it resets to 0 and adds that timestamp. It returns a DatetimeIndex of all event timestamps.

**Why CUSUM is superior to alternatives:**

Compare it to Bollinger Bands or fixed thresholds. With Bollinger Bands, an event triggers when price crosses a threshold. The problem is you get multiple false triggers when the price hovers around the threshold level.

CUSUM requires a cumulative deviation to exceed the threshold. It needs sustained movement - a full run of length h from the reset level. This filters out choppy sideways markets and captures actual directional moves.

One practical aspect that makes CUSUM filters appealing is that multiple events are NOT triggered by the price hovering around a threshold level, which is a flaw suffered by popular market signals like Bollinger Bands. CUSUM will require a full run of length h to trigger an event. This makes CUSUM robust to noise.

**Figure 2.3 illustration:**

The book shows a price series as a continuous line with sampled observations marked as dots. What you notice is more samples during volatile/trending periods (high information content) and fewer samples during quiet/sideways periods (low information). It's adaptive sampling based on actual market behavior.

**Alternative variables for CUSUM:**

You're not limited to price returns. You can apply CUSUM to structural break statistics from Chapter 17, entropy measures from Chapter 18, market microstructure metrics like VPIN or order flow toxicity from Chapter 19, or volatility measures like realized volatility or ATR.

For example, you could declare an event whenever the SADF statistic departs sufficiently from a previous reset level. Once you have this subset of event-driven bars, you let the ML algorithm determine whether the occurrence of such events constitutes actionable intelligence.

## Practical Implementation

**Step 1: Choose your metric** - Decide what variable should trigger sampling. It could be returns (prices.pct_change()), volatility (returns.rolling(20).std()), or any custom signal.

**Step 2: Set threshold** - The threshold h determines sensitivity. A small h (like 0.5%) gives you more events but more noise. A large h (like 2%) gives you fewer events but stronger signals. A good starting point is h = 1 standard deviation of your metric.

**Step 3: Apply CUSUM filter** - Use the getTEvents function to get event timestamps, then extract sampled bars at those timestamps.

**Step 4: Build features** - Use the sampled bars for ML training. Compute features at event times and labels (using methods from Chapter 3).

## Key Advantages

**Adaptive to market conditions:** CUSUM automatically increases sampling during volatile periods and reduces it during quiet periods. No manual adjustment needed.

**Robust to noise:** It requires sustained movement to trigger. It filters out random fluctuations and avoids false signals from choppy markets.

**Theoretically grounded:** Based on statistical quality control with well-understood mathematical properties. It's been used in industrial processes for decades.

**Computationally efficient:** It's an online algorithm that processes one bar at a time. You don't need to store the entire history, making it suitable for real-time applications.

**Flexible:** Works with any metric (prices, volatility, entropy, etc.). The threshold can be adjusted for your strategy needs, and it can detect both upward and downward moves.

## Common Pitfalls

**Threshold too small:** You get too many events, including noise. Solution: increase h and calibrate using cross-validation.

**Threshold too large:** You miss important events. Solution: decrease h and monitor event frequency.

**Wrong metric:** Events don't align with your strategy. Solution: experiment with different variables like volatility or microstructure metrics.

**Look-ahead bias:** Using future information in event detection. Solution: ensure E_{t-1}[y_t] uses only past data.

**Overfitting threshold:** Optimizing h on the same data used for backtest. Solution: use a separate calibration period or walk-forward analysis.

## Integration with ML Pipeline

The complete workflow is: First, data ingestion (collect bars, clean and validate). Second, event detection using CUSUM (apply filter to metric, generate event timestamps, sample relevant bars). Third, feature engineering (compute features at event times, include pre-event context, add microstructure features). Fourth, labeling using methods from Chapter 3 like triple-barrier. Fifth, model training using only sampled observations with proper cross-validation from Chapter 7. Sixth, backtesting on out-of-sample events.

## Summary

The problem is too many bars, most containing noise, creating computational burden and overfitting risk.

The solution is the CUSUM filter - a quality-control method that accumulates deviations from expected value, triggers events when threshold exceeded, resets after event (preventing cascading triggers), and adapts to market conditions automatically.

The benefit is building a smaller, higher-quality training set where each observation represents a significant market event, leading to better ML model performance, reduced computational requirements, less overfitting, and more interpretable results.

The philosophy is event-driven machine learning - let the market tell you when to pay attention, rather than sampling arbitrarily by time or randomly. This represents a fundamental shift from traditional time-based approaches to information-based approaches in quantitative finance.

Portfolio managers typically place a bet after some event takes place. If you ask a classifier to predict the sign of the next 5% absolute return after certain catalytic conditions are met, you're more likely to find informative features that will help achieve more accurate prediction.


