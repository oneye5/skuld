"""Run full model evaluation with rolling windows.

This is the main entry point for evaluating the ranking model's performance
across multiple time windows with proper train/test separation.

Usage:
    uv run python scripts/run_model_evaluation.py
    uv run python scripts/run_model_evaluation.py --forward-days 20
    uv run python scripts/run_model_evaluation.py --num-windows 3 --top-n 5
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging_config import setup_logging
from pipeline.ranking_pipeline import (
    run_ranking_pipeline, 
    print_ranking_summary,
    RankingPipelineResult,
)
from config.settings import (
    FORWARD_RETURN_DAYS,
    PORTFOLIO_TOP_N,
    PORTFOLIO_BOTTOM_N,
    TRANSACTION_COST_BPS,
    SLIPPAGE_BPS,
    MIN_STOCKS_PER_TIMESTAMP,
    NUM_ROLLING_WINDOWS,
    ROLLING_WINDOW_MOVEMENT_YEARS,
    TEST_PERIOD_YEARS,
)


def main():
    # Initialize logging - outputs to console by default
    setup_logging(level=logging.INFO, console=True)
    
    parser = argparse.ArgumentParser(
        description="Run ranking-based stock prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Window settings
    parser.add_argument(
        "--num-windows", type=int, default=NUM_ROLLING_WINDOWS,
        help="Number of rolling windows"
    )
    parser.add_argument(
        "--window-movement", type=float, default=ROLLING_WINDOW_MOVEMENT_YEARS,
        help="Years between window starts"
    )
    parser.add_argument(
        "--test-period", type=float, default=TEST_PERIOD_YEARS,
        help="Test period length in years"
    )
    
    # Target settings
    parser.add_argument(
        "--forward-days", type=int, default=FORWARD_RETURN_DAYS,
        help="Forward return horizon in days"
    )
    parser.add_argument(
        "--return-type", choices=["simple", "log"], default="simple",
        help="Type of return calculation"
    )
    parser.add_argument(
        "--no-winsorize", action="store_true",
        help="Disable winsorization of returns"
    )
    
    # Portfolio settings
    parser.add_argument(
        "--top-n", type=int, default=PORTFOLIO_TOP_N,
        help="Number of stocks for long portfolio"
    )
    parser.add_argument(
        "--bottom-n", type=int, default=PORTFOLIO_BOTTOM_N,
        help="Number of stocks for short portfolio"
    )
    parser.add_argument(
        "--cost-bps", type=float, default=TRANSACTION_COST_BPS,
        help="Transaction cost in basis points"
    )
    parser.add_argument(
        "--slippage-bps", type=float, default=SLIPPAGE_BPS,
        help="Slippage in basis points (market impact, bid-ask spread)"
    )
    
    # Other settings
    parser.add_argument(
        "--min-stocks", type=int, default=MIN_STOCKS_PER_TIMESTAMP,
        help="Minimum stocks per timestamp"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to output directory"
    )
    
    args = parser.parse_args()
    
    # Determine winsorize limits
    winsorize_limits = None if args.no_winsorize else (-0.5, 0.5)
    
    print("=" * 60)
    print("RANKING PIPELINE")
    print("=" * 60)
    print(f"Forward return days:  {args.forward_days}")
    print(f"Return type:          {args.return_type}")
    print(f"Number of windows:    {args.num_windows}")
    print(f"Portfolio top-N:      {args.top_n}")
    print(f"Portfolio bottom-N:   {args.bottom_n}")
    print(f"Transaction cost:     {args.cost_bps} bps")
    print(f"Slippage:             {args.slippage_bps} bps")
    print(f"Min stocks/timestamp: {args.min_stocks}")
    print("=" * 60)
    
    # Run pipeline
    result = run_ranking_pipeline(
        num_windows=args.num_windows,
        window_movement_years=args.window_movement,
        test_period_years=args.test_period,
        forward_return_days=args.forward_days,
        return_type=args.return_type,
        winsorize_limits=winsorize_limits,
        min_stocks=args.min_stocks,
        portfolio_top_n=args.top_n,
        portfolio_bottom_n=args.bottom_n,
        transaction_cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        save_results=not args.no_save,
    )
    
    # Print summary
    print_ranking_summary(result)
    
    return result


if __name__ == "__main__":
    main()
