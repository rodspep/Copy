"""UG reverse-engineering analysis: signal model + feature/validator engine.

Given captured UG signals and OHLC (TradingView export — the authoritative feed
per tv_loader, or MT5 parquet), build a feature matrix of indicator/structure
conditions at each signal's bar. Downstream stats then find which combination of
conditions consistently coincides with UG entries → UG's logic.

Everything is lookahead-safe: indicators are causal (value at bar i uses only
bars ≤ i) and signals read the last CLOSED bar at-or-before their timestamp.
"""
