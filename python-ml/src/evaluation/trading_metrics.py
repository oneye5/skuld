from dataclasses import dataclass
from typing import Dict


@dataclass
class TradingMetrics:
    """Container for trading performance metrics."""
    # --- Volume ---
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # --- Returns ---
    total_return: float
    avg_return: float
    median_return: float
    std_return: float
    best_trade: float
    worst_trade: float

    # --- Distribution ---
    return_25th: float
    return_75th: float
    return_iqr: float
    skewness: float
    kurtosis: float

    # --- Profitability ---
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    expectancy: float

    # --- System Quality ---
    sqn: float  # System Quality Number
    kelly_criterion: float

    # --- Streaks ---
    max_consecutive_wins: int
    max_consecutive_losses: int

    # --- Risk Adjusted ---
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # --- Drawdown & Tail Risk ---
    max_drawdown: float
    max_drawdown_duration: int
    recovery_factor: float
    ulcer_index: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR / Expected Shortfall (95%)

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {k: v for k, v in self.__dict__.items()}
