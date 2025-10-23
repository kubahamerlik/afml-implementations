# Financial Data Structures - Chapter 1 Notes

## Why Time Bars Are Problematic

**Time bars** sample at fixed intervals (e.g., every 1 minute), collecting:
- Timestamp, VWAP, OHLC, Volume

**Two key problems:**

1. **Ignores market activity rhythm**
   - Markets don't process information at constant intervals
   - Hour after open = very active
   - Noon hour = quiet
   - Result: oversampling during quiet periods, undersampling during volatile periods

2. **Poor statistical properties**
   - **Serial correlation**: current prices depend on past values (not independent)
   - **Heteroscedasticity**: variance changes over time (not constant)
   - Non-normal return distribution

**Mandelbrot & Taylor insight:**
Time bars mix TWO random processes:
- Price changes per transaction (normal distribution)
- Number of transactions per time period (stable Paretian)
- Result = stable Paretian with fat tails and infinite variance

**Solution:** Sample based on trading activity, not time

---

## Standard Activity-Based Bars

### Comparison Table

| Type | Trigger | Advantage | Limitation |
|------|---------|-----------|------------|
| **Time** | Fixed time interval | Simple, standard | Ignores activity; poor statistics |
| **Tick** | Fixed # transactions | Adapts to activity | Treats all trades equally |
| **Volume** | Fixed # shares/contracts | Considers trade size | Ignores price changes |
| **Dollar** | Fixed $ value | Considers size AND price | Not time-based |

### Tick Bars

**Trigger:** Every N transactions (e.g., 1,000 ticks)

**Advantages:**
- Synchronizes with information arrival
- Better statistical properties than time bars

**Caution:** Opening/closing auctions create outliers
- Thousands of transactions execute simultaneously
- Filter or handle separately

### Volume Bars

**Trigger:** Every N shares traded (e.g., 10,000 shares)

**Advantages vs Tick:**
- 1,000-share order ≠ 1-share order
- Captures institutional activity better

**Limitation:** 1,000 shares at $10 ≠ 1,000 shares at $500

### Dollar Bars

**Trigger:** Every $N traded (e.g., $1M = price × volume)

**Advantages:**
- Accounts for both volume AND price
- Best statistical properties (closest to IID Gaussian)
- Recommended by López de Prado for ML

**Use case:** Multi-asset portfolios, long-term analysis

---

## Information-Driven Bars

**Core idea:** Sample when **informed traders** enter the market

**Goal:** Make decisions BEFORE prices reach new equilibrium

**Two types:**
- **Imbalance Bars**: Detect net buying/selling pressure
- **Run Bars**: Detect sustained directional sequences

---

## Imbalance Bars

**Concept:** Close bar when cumulative imbalance exceeds expected level

### How Imbalance Works

**Step 1: Label each trade**
- Price UP from last trade → **+1** (buyer initiated)
- Price DOWN from last trade → **-1** (seller initiated)

**Step 2: Calculate cumulative imbalance**
- Add +1s and -1s together
- Example: +1, +1, -1, +1 = +2 (net buying)

**Step 3: Close bar when unusual**
- If total far from zero → close bar

### Imbalance Bar Types

| Type | Weighs By | Formula | Best For |
|------|-----------|---------|----------|
| **Tick Imbalance** | Direction only | Sum of ±1 | Quick reaction (noisy) |
| **Volume Imbalance** | Trade size | Sum of (±1 × volume) | Institutional detection |
| **Dollar Imbalance** | Economic value | Sum of (±1 × price × volume) | Multi-asset, capital flows |

**Why this works:**
- Balanced market → imbalance ≈ 0 → few bars
- Informed trading → large imbalance → many bars

---

## Run Bars

**Concept:** Close bar when consecutive same-direction trades exceed expected length

### How Runs Work

**What is a "run"?**
- BUY, BUY, BUY, BUY = run of 4 buys
- SELL, SELL, SELL = run of 3 sells
- Direction change breaks the run

**Process:**
1. Count consecutive buys or sells
2. Compare to expected run length
3. Close bar when run unusually long

### Run Bar Types

| Type | Weighs By | Best For |
|------|-----------|----------|
| **Tick Runs** | Count only | Sustained sequences |
| **Volume Runs** | Trade size | Institutional directional moves |
| **Dollar Runs** | Economic value | Capital commitment direction |

**Why this works:**
- Choppy market: BUY, SELL, BUY, SELL → short runs → few bars
- Directional move: BUY, BUY, BUY, BUY, BUY, BUY → long run → many bars

---

## Imbalance vs Runs: Key Difference

**Imbalance Bars:**
- Measure: total buys - total sells
- Can trigger with alternating directions
- Example: +1, -1, +1, +1, -1, +1 = +2 → triggers bar

**Runs Bars:**
- Measure: length of consecutive sequence
- Require sustained one-direction flow
- Example: +1, -1, +1, +1, -1, +1 = short runs → NO bar
- Need: +1, +1, +1, +1, +1, +1 → triggers bar

**Best practice:** Use both - they capture different aspects of informed trading

---

## Quick Reference

**Choosing bar type:**
- Starting out → **Time bars** (simple)
- Activity-based → **Tick/Volume bars**
- ML/quant → **Dollar bars** (best statistics)
- Detecting informed traders → **Dollar Imbalance + Dollar Runs**

**Key definitions:**
- **VWAP**: Volume-weighted average price = Σ(price × volume) / Σ(volume)
- **Serial correlation**: Current value depends on past values
- **Heteroscedasticity**: Variance changes over time
- **IID**: Independent and identically distributed (ML assumption)
