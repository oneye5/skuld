# Skuld Ranking Pipeline - TODO List

**Last Updated**: 2025-12-29  
**Purpose**: Track improvements and bug fixes for the ranking-based stock prediction pipeline.

---

## ✅ Completion Tracker

| # | Task | Priority | Status | Completed Date |
|---|------|----------|--------|----------------|
| 1 | Fix forward return tolerance | HIGH | ⬜ | |
| 2 | Fix long_only config mismatch | HIGH | ⬜ | |
| 3 | Add train/test gap = forward_days | HIGH | ⬜ | |
| 4 | Investigate numpy overflow warning | MEDIUM-HIGH | ⬜ | |
| 5 | Filter out extremely illiquid stocks | MEDIUM-HIGH | ⬜ | |
| 6 | Add staleness features | MEDIUM | ⬜ | |
| 7 | Add dollar volume rank features | MEDIUM | ⬜ | |
| 8 | Investigate dropped features warning | MEDIUM | ⬜ | |
| 9 | Tune tree depth to 3-6 leaves | MEDIUM | ⬜ | |
| 10 | Add early stopping with validation | MEDIUM | ⬜ | |
| 11 | Add Momentum × Volatility interactions | MEDIUM | ⬜ | |
| 12 | Add interest rate × feature interactions | MEDIUM | ⬜ | |
| 13 | Add interest rate × debt-to-equity interaction | MEDIUM | ⬜ | |
| 14 | Set add_missing_flags=True | LOW-MEDIUM | ⬜ | |
| 15 | Use rank_xendcg objective | LOW-MEDIUM | ⬜ | |
| 16 | Add defensive fundamental ratios | LOW | ⬜ | |
| 17 | Investigate using adjusted close | LOW | ⬜ | |

**Legend**: ⬜ Not Started | 🔄 In Progress | ✅ Complete | ❌ Blocked

---

## 🔴 Critical Issues (Fix First)

### Forward Return Tolerance Too Permissive
- **Issue**: `compute_forward_returns()` defaults `tolerance_days = lookahead_days // 2 + 5`
  - For 365-day horizon, tolerance is 192 days
  - This silently converts "1Y forward return" into "~1Y to ~1.5Y forward return"
  - Increases label noise and biases results on illiquid/stale series
- **Location**: [core/target_builder.py](../python/ml-pipeline/core/target_builder.py)
- **Solution**: 
  - Reduce tolerance to 10-20 days max for 365-day horizon
  - Add validation warning when tolerance > 30 days
  - Consider making tolerance a fixed value rather than function of lookahead
- **Priority**: HIGH

### Config Mismatch: Long-Only Setting Not Propagated
- **Issue**: `long_only=True` set in `config/settings.py` but not used in `PortfolioConfig`
- **Impact**: Portfolio simulator may short stocks despite long-only configuration
- **Location**: 
  - [config/settings.py](../python/ml-pipeline/config/settings.py)
  - [evaluation/portfolio_simulator.py](../python/ml-pipeline/evaluation/portfolio_simulator.py)
  - [scripts/run_model_evaluation.py](../python/ml-pipeline/scripts/run_model_evaluation.py)
- **Solution**: 
  - Update pipeline to respect `LONG_ONLY` setting from config
  - Set `bottom_n=0` when `long_only=True`
- **Priority**: HIGH

### Numpy Overflow Warning
- **Issue**: `RuntimeWarning: overflow encountered in accumulate` in numpy fromnumeric.py
- **Location**: `D:\Projects\StandAloneProjects\skuld\python\ml-pipeline\.venv\Lib\site-packages\numpy\_core\fromnumeric.py:54`
- **Impact**: May corrupt cumulative calculations (returns, positions, etc.)
- **Investigation Needed**:
  - Identify which accumulate operations trigger overflow
  - Check if this occurs in portfolio backtest or feature engineering
  - Use float64 for cumulative calculations or clip extreme values
- **Priority**: MEDIUM-HIGH

---

## 🟡 Model Configuration & Training

### Add Train/Test Gap = Forward Days
- **Issue**: Current rolling windows may have overlapping forward returns between train/test
- **Solution**: Add gap between train end and test start equal to `forward_days`
  - Ensures no label leakage from train to test
  - Window structure: `[train] -> [gap=forward_days] -> [test]`
- **Location**: [core/splitter.py](../python/ml-pipeline/core/splitter.py)
- **Priority**: HIGH

### Tune Tree Depth (Max Leaves)
- **Current**: Default LightGBM `num_leaves=31`
- **Proposed**: Reduce to 3-6 leaves for better generalization
- **Rationale**: 
  - Financial markets are noisy; simpler trees prevent overfitting
  - Cross-sectional ranking benefits from linear-like structure
- **Location**: [learner/ranking.py](../python/ml-pipeline/learner/ranking.py), `RankerConfig`
- **Action**: 
  - Test with `num_leaves=[3, 5, 7]`
  - Compare IC/ICIR across configurations
- **Priority**: MEDIUM

### Add Early Stopping with Validation
- **Current**: Fixed `n_estimators=100`
- **Proposed**: Use early stopping on validation set
- **Benefits**: 
  - Prevents overfitting
  - Adaptive model complexity per window
- **Implementation**:
  - Reserve 20% of train data as validation set
  - Use `early_stopping_rounds=20` in LGBMRanker
- **Location**: [learner/ranking.py](../python/ml-pipeline/learner/ranking.py)
- **Priority**: MEDIUM

### Use Rank XENDCG Objective
- **Current**: Default LGBMRanker objective (likely `lambdarank`)
- **Proposed**: Experiment with `rank_xendcg` objective
- **Rationale**: May better optimize for top-N selection
- **Location**: [learner/ranking.py](../python/ml-pipeline/learner/ranking.py), `RankerConfig`
- **Action**: Compare performance vs. current objective
- **Priority**: LOW-MEDIUM

---

## 🟢 Feature Engineering

### Feature Interactions

#### 1. Momentum × Volatility Interactions
- **Examples**:
  - `Mom_20d × Vol_20d` (high momentum + low vol = quality)
  - `Mom_252d / Vol_252d` (risk-adjusted momentum)
- **Location**: [features/cross_sectional.py](../python/ml-pipeline/features/cross_sectional.py) or new `features/interactions.py`

#### 2. Interest Rate × Feature Interactions
- **Examples**:
  - `Interest_Rate × Debt_to_Equity` (rate sensitivity)
  - `Interest_Rate × Vol_20d` (rate regime volatility)
  - `Interest_Rate × Mom_252d` (rate cycle momentum)
- **Rationale**: Stock behavior changes with rate environment
- **Prerequisites**: Ensure interest rate macro data is available

#### 3. Interest Rate × Debt-to-Equity
- **Specific Case**: High-debt companies more sensitive to rate changes
- **Feature**: `IR_DebtSensitivity = InterestRate * DebtToEquity`
- **Location**: [features/interactions.py](../python/ml-pipeline/features/interactions.py) (new module)

**Priority**: MEDIUM (interactions can significantly improve model)

### Dollar Volume Rank Features
- **Feature**: Cross-sectional rank of `Close × Volume` at each timestamp
- **Rationale**: Liquidity signal independent of absolute price
- **Implementation**:
  ```python
  df['DollarVolume'] = df['Close'] * df['Volume']
  df['Rank_DollarVolume'] = df.groupby(TIMESTAMP)['DollarVolume'].rank(pct=True)
  ```
- **Location**: [features/cross_sectional.py](../python/ml-pipeline/features/cross_sectional.py)
- **Priority**: MEDIUM

### Defensive Fundamental Ratios
- **Add**:
  - `Debt_to_Equity` (leverage risk)
  - `Current_Ratio` (liquidity)
  - `Interest_Coverage` (debt service ability)
- **Challenge**: These require financial statement data (not in current dataset)
- **Workaround**: Proxy with volatility/volume metrics or extend data source
- **Location**: [features/ratios.py](../python/ml-pipeline/features/ratios.py)
- **Priority**: LOW (data availability issue)

### Staleness Features
- **Rationale**: Stale prices (no trading) are a quality/liquidity signal
- **Features**:
  - `Days_Since_Volume_Change` (consecutive days with same volume)
  - `Days_Since_Price_Change` (consecutive days with same close)
  - `Price_Staleness_Flag` (binary: price unchanged for N days)
- **Location**: [features/technical.py](../python/ml-pipeline/features/technical.py)
- **Priority**: MEDIUM

---

## 🔵 Data Quality & Preprocessing

### Filter Out Extremely Illiquid Stocks
- **Criterion**: 
  - Stocks with Volume < threshold (e.g., $10K daily dollar volume)
  - Or: stocks with >50% zero-volume days in trailing window
- **Rationale**: Illiquid stocks have unreliable returns and are untradeable
- **Location**: [core/data_loader.py](../python/ml-pipeline/core/data_loader.py) or [pipeline/ranking_pipeline.py](../python/ml-pipeline/pipeline/ranking_pipeline.py)
- **Implementation**:
  ```python
  df['DollarVolume'] = df['Close'] * df['Volume']
  median_dv = df.groupby(TICKER)['DollarVolume'].median()
  liquid_tickers = median_dv[median_dv > 10_000].index
  df = df[df[TICKER].isin(liquid_tickers)]
  ```
- **Priority**: MEDIUM-HIGH

### Set `add_missing_flags=True`
- **Current**: `add_missing_flags=False` in `preprocess_data()`
- **Proposed**: Enable missing value flags as features
- **Rationale**: Missing patterns can be informative (e.g., no volume = illiquid)
- **Location**: [core/preprocessor.py](../python/ml-pipeline/core/preprocessor.py)
- **Change**: Update default or pass `add_missing_flags=True` in pipeline
- **Priority**: LOW-MEDIUM

### Investigate Using Adjusted Close
- **Issue**: Current data uses raw `Close` prices (no dividend adjustment)
- **Impact**: 
  - Overstates returns on ex-dividend dates
  - Biases momentum/return features
- **Challenge**: Adjusted close not available in current dataset (`data/data_long.csv`)
- **Options**:
  1. Extend data source to include adjusted close
  2. Manual adjustment using dividend/split events (if available)
  3. Accept limitation and document
- **Priority**: LOW (data availability issue)

---

## 🟣 Investigation & Debugging

### Dropped Features Warning
- **Issue**: Consistent warning: "Dropped 60 features present in train but not test"
- **Example Features**: `Rev_5d_Skip1`, `AvgQuarterRet_252d`, `PosQuarters_252d`, `Rank_RelVol_20d`, `Trend_RSq_20`
- **Root Cause**: 
  - Features have insufficient data in test set (all NaN or sparse)
  - Rolling window lookback extends beyond available history in test period
- **Investigation Steps**:
  1. Log which features are dropped and why
  2. Check if dropped features are recent additions or complex lag features
  3. Adjust feature construction to ensure test coverage
  4. Consider stricter sparsity threshold in training
- **Location**: [pipeline/ranking_pipeline.py](../python/ml-pipeline/pipeline/ranking_pipeline.py) (feature alignment logic)
- **Priority**: MEDIUM

---

## 📋 Implementation Priority Matrix

| Task | Priority | Effort | Impact | Dependencies |
|------|----------|--------|--------|--------------|
| Fix forward return tolerance | HIGH | Low | High | None |
| Add train/test gap | HIGH | Medium | High | None |
| Fix long_only config mismatch | HIGH | Low | Medium | None |
| Filter illiquid stocks | MEDIUM-HIGH | Low | High | None |
| Investigate numpy overflow | MEDIUM-HIGH | Medium | High | None |
| Add dollar volume rank | MEDIUM | Low | Medium | None |
| Add staleness features | MEDIUM | Medium | Medium | None |
| Tune tree depth | MEDIUM | Low | Medium | None |
| Add early stopping | MEDIUM | Medium | Medium | None |
| Add Mom×Vol interactions | MEDIUM | Medium | High | None |
| Add IR × feature interactions | MEDIUM | Medium | High | Macro data |
| Investigate dropped features | MEDIUM | High | Medium | None |
| Set add_missing_flags=True | LOW-MEDIUM | Low | Low | None |
| Use rank_xendcg | LOW-MEDIUM | Low | Medium | None |
| Add defensive ratios | LOW | High | High | Financial data |
| Investigate adjusted close | LOW | High | Medium | Data source |

---

## 🎯 Suggested Implementation Sequence

### Phase 1: Critical Fixes (Week 1)
1. ✅ Fix forward return tolerance (HIGH priority, low effort)
2. ✅ Fix long_only config mismatch (HIGH priority, low effort)
3. ✅ Add train/test gap (HIGH priority, medium effort)
4. ✅ Investigate numpy overflow (MEDIUM-HIGH priority)

### Phase 2: Data Quality (Week 2)
5. ✅ Filter illiquid stocks (MEDIUM-HIGH priority)
6. ✅ Add staleness features (MEDIUM priority)
7. ✅ Add dollar volume rank (MEDIUM priority)
8. ✅ Investigate dropped features warning (MEDIUM priority)

### Phase 3: Model Improvements (Week 3)
9. ✅ Tune tree depth (MEDIUM priority)
10. ✅ Add early stopping (MEDIUM priority)
11. ✅ Experiment with rank_xendcg (LOW-MEDIUM priority)

### Phase 4: Feature Engineering (Week 4)
12. ✅ Add Mom×Vol interactions (MEDIUM priority)
13. ✅ Add IR × feature interactions (MEDIUM priority)
14. ✅ Set add_missing_flags=True (LOW-MEDIUM priority)

### Phase 5: Future Enhancements
15. 🔲 Add defensive fundamental ratios (LOW priority, data dependent)
16. 🔲 Investigate adjusted close (LOW priority, data dependent)

---

## 📝 Notes

- **Before implementing**: Review [docs/RANKING_PIPELINE_GUIDE.md](RANKING_PIPELINE_GUIDE.md) for context
- **After each change**: Run `uv run pytest` to ensure no regressions
- **Performance validation**: Use `scripts/run_model_evaluation.py --num-windows 3` for quick validation
- **Experiment tracking**: Use `core/experiment_tracking.py` to track all configuration changes

---

## 🔗 Related Documentation

- [RANKING_PIPELINE_GUIDE.md](RANKING_PIPELINE_GUIDE.md) - Main pipeline documentation
- [config/settings.py](../python/ml-pipeline/config/settings.py) - Configuration defaults
- [tests/](../python/ml-pipeline/tests/) - Unit tests for validation

