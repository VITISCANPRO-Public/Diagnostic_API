"""
Root-level conftest.py — marks the project root for pytest.

This file ensures that pytest adds the project root to sys.path,
allowing test files to import application modules directly
(e.g. 'from app import app').

No shared fixtures are defined here for now.
"""