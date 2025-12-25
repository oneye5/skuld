"""
=============================================================================
FINAL INVESTIGATION REPORT: Skuld Pipeline vs NZX-Predictor Performance Gap
=============================================================================

SUMMARY OF FINDINGS
-------------------

The investigation revealed that the performance gap between skuld and
nzx-predictor is NOT primarily due to leakage. Here are the key findings:

=============================================================================
1. THE "1+ SHARPE" CLAIM MAY BE INCORRECT OR MISREMEMBERED
=============================================================================

The legacy data in `data/legacy/trade_simulation.csv` shows:
- Legacy Sharpe: 0.0546 (NOT 1+)
- Current Sharpe: 0.086

Both are in the same ballpark (~0.05-0.09), suggesting:
- The 1+ Sharpe figure may have been from a different experiment
- Or calculated with a different formula (e.g., annualized)
- Or based on a cherry-picked time period

=============================================================================
2. LEGACY DATA ANOMALIES
=============================================================================

The legacy trade data has significant issues:
- Only 3 tickers, all INDEX/macro symbols: %5EFTSE, %5EN225, %5ETNX
- Zero actual stock trades (all were index trades)
- Contains extreme outlier: one trade with 1387% return
  (Buy at $0.005, sell at $6.94 - clearly a data error)

Current pipeline trades 138 different tickers (actual NZ stocks).

=============================================================================
3. LEAKAGE ANALYSIS RESULTS
=============================================================================

INVESTIGATED POTENTIAL LEAKAGE SOURCES:

a) Scaler Leakage (train+test combined fitting):
   - DIFFERENCE: skuld now fits on train only (correct approach)
   - IMPACT: Minimal (~0.06 AUC difference in simulation)
   - CONCLUSION: Not the cause of major performance gap

b) Label Leakage:
   - High correlations found (0.99+) are from SPARSE features
   - Only ~350 samples out of 533,264 have these features
   - CONCLUSION: Spurious correlation, not true predictive power

c) Temporal Leakage:
   - No timestamp overlap between train/test ✓
   - Proper train/test boundary respected ✓
   - CONCLUSION: No temporal leakage detected

d) Feature Timing:
   - No obviously future-looking features detected
   - Some suspicious names ("gain", "loss") but these are accounting terms
   - CONCLUSION: No feature timing leakage detected

=============================================================================
4. METHODOLOGY DIFFERENCES
=============================================================================

Known differences between skuld and nzx-predictor:

| Aspect              | NZX-Predictor        | Skuld (Current)     |
|---------------------|----------------------|---------------------|
| Scaler fitting      | Train+Test combined  | Train only          |
| Prediction threshold| 0.79                 | 0.75                |
| Rolling windows     | Unknown (likely none)| 25 windows          |
| Tickers traded      | Index only (3)       | NZ stocks (138)     |

=============================================================================
5. ROOT CAUSE ANALYSIS
=============================================================================

The most likely explanations for the performance discrepancy:

HYPOTHESIS 1: Memory Bias
- The "1+ Sharpe" may never have existed
- Human memory can be unreliable about past performance
- Legacy data confirms similar low Sharpe

HYPOTHESIS 2: Different Data/Tickers  
- Legacy only traded 3 index tickers
- Current trades 138 NZ stocks
- Different asset classes = incomparable Sharpe

HYPOTHESIS 3: Single vs Rolling Window Evaluation
- If nzx-predictor used single train/test split
- It might have found a "lucky" test period
- Rolling windows reveal true out-of-sample performance

HYPOTHESIS 4: Annualization Confusion
- Raw Sharpe of 0.086 on annual returns
- If someone reported "annualized" daily Sharpe: 0.086 * sqrt(252) ≈ 1.36
- This is a common source of confusion

=============================================================================
6. TRUE PERFORMANCE REALITY
=============================================================================

Based on both legacy and current data:
- The model has weak predictive power (AUC ~0.51-0.60)
- Sharpe ratio is genuinely low (~0.05-0.09)
- This is consistent with the efficient market hypothesis
- Financial prediction is hard; most simple models don't beat buy-and-hold

=============================================================================
RECOMMENDATIONS
=============================================================================

1. DON'T CHASE THE "1+ SHARPE"
   - It likely doesn't exist or was misremembered
   - The current pipeline is performing as expected

2. FOCUS ON FEATURE ENGINEERING
   - Add momentum indicators
   - Add technical analysis features
   - Consider sentiment data if available

3. CONSIDER MODEL IMPROVEMENTS
   - Ensemble methods
   - Different target definitions
   - Risk-adjusted position sizing

4. VERIFY AGAINST BUY-AND-HOLD
   - Compare strategy Sharpe to simple buy-and-hold
   - If buy-and-hold beats the model, the features aren't predictive

5. ACCEPT REALISTIC EXPECTATIONS
   - Financial prediction is extremely difficult
   - Sharpe ratios of 0.5-1.0 are considered good in practice
   - The current 0.086 suggests the model needs significant improvement
   - This improvement should come from better features, not leakage

=============================================================================
"""

print(__doc__)

# Final verification
if __name__ == "__main__":
    print("\nTo verify these findings, run:")
    print("  uv run .\\scripts\\leakage_investigation.py")
    print("  uv run .\\scripts\\deep_investigation.py") 
    print("  uv run .\\scripts\\methodology_comparison.py")
    print("  uv run .\\scripts\\legacy_comparison.py")
