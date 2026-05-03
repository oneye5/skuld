"""Tests for deflated Sharpe ratio."""
from __future__ import annotations

from skuld_research.stats.deflated import deflated_sharpe


def test_deflated_falls_with_more_trials():
    """Deflated Sharpe decreases monotonically as n_trials grows."""
    sharpe_hat = 2.0
    n_obs = 240

    d1 = deflated_sharpe(sharpe_hat, n_obs, n_trials=1)
    d10 = deflated_sharpe(sharpe_hat, n_obs, n_trials=10)
    d100 = deflated_sharpe(sharpe_hat, n_obs, n_trials=100)
    d1000 = deflated_sharpe(sharpe_hat, n_obs, n_trials=1000)

    assert d1.sharpe_deflated > d10.sharpe_deflated
    assert d10.sharpe_deflated > d100.sharpe_deflated
    assert d100.sharpe_deflated > d1000.sharpe_deflated


def test_noise_floor_fails_at_high_prior():
    """Sharpe at noise floor fails gating when n_trials_prior is large."""
    sharpe_hat = 0.5  # weak Sharpe
    n_obs = 120
    n_trials = 30

    result = deflated_sharpe(sharpe_hat, n_obs, n_trials, alpha=0.05)
    assert result.passes == False


def test_strong_sharpe_passes_with_few_trials():
    """Strong Sharpe with n_trials=1 passes gating."""
    sharpe_hat = 3.0
    n_obs = 240
    n_trials = 1

    result = deflated_sharpe(sharpe_hat, n_obs, n_trials, alpha=0.05)
    assert result.passes == True
    assert result.sharpe_deflated > 0


def test_negative_sharpe_never_passes():
    """Negative Sharpe always fails gating."""
    result = deflated_sharpe(-1.0, 100, 1, alpha=0.05)
    assert result.passes == False


def test_zero_sharpe():
    """Zero Sharpe returns deflated=negative (penalised by expected max)."""
    result = deflated_sharpe(0.0, 100, 10, alpha=0.05)
    # deflated should be negative since we subtract expected_max_SR
    assert result.sharpe_deflated < 0
    assert result.passes == False
