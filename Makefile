.PHONY: verify mutants

# Fast loop: meta gate + assertion-strength gate + non-integration/system tests.
verify:
	uv run pytest -q -m "not integration and not system"

# On-demand mutation testing over the pure, stateless modules (data/harmonise.py,
# eval/ccc.py, video_encoding/mesh/quorum.py, video_encoding/mesh/outer_step.py).
# Scope and test selection live in pyproject.toml's [tool.mutmut] section.
# `uv run mutmut results` lists survivors after this finishes.
mutants:
	uv run mutmut run
