.PHONY: verify mutants

# Fast loop: every test except the integration and system ones.
verify:
	uv run pytest -q -m "not integration and not system"

# On-demand mutation testing over the pure, stateless modules (stern/data/harmonise.py,
# stern/evaluation/ccc.py, stern/mesh/quorum.py, stern/mesh/outer_step.py).
# Scope and test selection live in pyproject.toml's [tool.mutmut] section.
# `uv run mutmut results` lists survivors after this finishes.
mutants:
	uv run mutmut run
