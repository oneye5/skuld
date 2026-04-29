"""Skuld portfolio — execution planner.

Computes the minimal set of trades (buys, sells, partial trims) needed
to move from a current Sharesies portfolio to the target allocation,
respecting lot constraints and the Sharesies fee cliff.
"""
