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
