"""Velora local-first agent package.

Two selectable run modes (read from the MODE env var):
  * "zero"   - stay fully local, never call the remote API (0 scored tokens).
  * "hybrid" - local-first, but escalate still-uncertain tasks to Fireworks.
"""
