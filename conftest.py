"""Pytest bootstrap.

Its mere presence at the repo root puts the root on ``sys.path`` during
collection, so the top-level ``api`` package (which is not pip-installed,
unlike ``airbnb_iip`` under ``src/``) is importable when CI runs bare
``pytest``.
"""
