"""TB-RT30: real-time (30-second) derivative of TB-Engine-v1 for the web server (docs/web-ser5.md).

Nothing here edits a frozen engine file. Two kinds of change are kept apart:

- behaviour-preserving speed-ups (``fast_windows``): with them installed, TB-Engine-v1 makes the
  same moves with the same diagnostics, nodes and calls ("TB1-fast"); checked by
  ``scripts/check_rt30_equivalence.py`` and the tests;
- the wall-clock deadline (``deadline``, ``vcf``, ``solver``): a cut becomes UNKNOWN through the
  same exception path as a node-budget cut, so it can lose a proof but never invent one. With
  ``deadline=None`` the solvers are the frozen ones, node for node.
"""
