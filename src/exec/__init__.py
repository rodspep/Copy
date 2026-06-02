"""Live execution: UG signal copier (filter → place limit → manage) for MT5.

DRY-RUN by default. Real order placement is gated behind an explicit --live flag
in scripts/ug_copier.py; the decision logic here is pure and unit-tested.
"""
