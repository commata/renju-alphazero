"""Offline analysis tools (threat proofs, tactical labels).

Nothing under ``search``, ``training`` or ``model`` may import this package: the
solvers here label and measure positions, they are never search rules
(``tests/test_analysis_isolation.py``).
"""
