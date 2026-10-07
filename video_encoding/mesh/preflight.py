"""Pre-launch roster screening: which requested nodes will actually train.

A pool can seed perfectly and still train on a fraction of its roster.
`epoch_advance.eligible_nodes` drops anything sitting in the watcher's
*global* `excluded_nodes`, and shard coverage is then recomputed over the
survivors and reported as a note on a log line -- after the 223 GB seed has
already run. cars-real10r and fish-real10r both passed validate_seeding at
full coverage and trained at 246/260 and 204/260 for exactly this reason.

Everything here is pure: callers supply the probe results. Keeping it free
of watcher imports preserves the layering (watcher imports mesh, never the
reverse) and lets every branch be exercised from a dict.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

# login-node is the login node and node089 runs the watcher cron; both appear
# in, or adjacent to, a tier roster (node089 is a real orchids row in
# tiers.tsv) and neither may ever be handed out as a pool node.
DENYLISTED_NODES = frozenset({"node089", "login-node"})

# Reasons exclude_node writes that name the run they came from. Only these
# may be reclaimed automatically -- a "manual:" flag is a human's judgement
# about the hardware and outlives any run, even though it also carries an
# "of {run_id}" suffix.
_RUN_SCOPED_REASONS = (
    re.compile(r"^no shards left to assign, of (?P<run_id>.+)$"),
    re.compile(r"^crash-loop: missed \d+ consecutive rounds of (?P<run_id>.+)$"),
)


@dataclass(frozen=True)
class RosterVerdict:
    """`rejected` is (node, reason) in the order the nodes were requested."""

    usable: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        # An empty roster is not ok: refusing to call a pool with no nodes
        # launchable mirrors validate_seeding's stance on an empty --nodes.
        return bool(self.usable) and not self.rejected


def screen_roster(
    requested: Sequence[str],
    *,
    excluded_nodes: Mapping[str, str],
    gpu_free_mib: Mapping[str, int | None] | None,
    min_gpu_free_mib: int,
    denylist: Collection[str] = DENYLISTED_NODES,
) -> RosterVerdict:
    """Split `requested` into nodes that will train and nodes that will not.

    `gpu_free_mib` of None skips the GPU check entirely, so an
    exclusions-only screen costs no ssh. A node present in neither the
    mapping nor the probe fails closed: an unanswered probe is never "fine".

    `min_gpu_free_mib` is required rather than defaulted so this module
    cannot drift from watcher.node_health.DEFAULT_MIN_GPU_FREE_MIB.
    """
    usable: list[str] = []
    rejected: list[tuple[str, str]] = []

    for node in dict.fromkeys(h.strip() for h in requested):
        if not node:
            continue

        if node in denylist:
            rejected.append((node, "denylisted: control node, never a pool node"))
            continue

        # Checked before the GPU: an exclusion is the reason a human has to
        # clear, and it is actionable without touching the node.
        if node in excluded_nodes:
            rejected.append((node, f"excluded: {excluded_nodes[node]}"))
            continue

        if gpu_free_mib is not None:
            free = gpu_free_mib.get(node)
            if free is None:
                rejected.append((node, "gpu unknown: probe failed"))
                continue
            if free < min_gpu_free_mib:
                rejected.append(
                    (node, f"gpu busy: {free} MiB free < {min_gpu_free_mib}")
                )
                continue

        usable.append(node)

    return RosterVerdict(usable=tuple(usable), rejected=tuple(rejected))


def stale_exclusions(
    excluded_nodes: Mapping[str, str], *, live_run_ids: Collection[str]
) -> dict[str, str]:
    """Exclusions owned by a run that no longer exists.

    `state.clear_run_state` only reclaims a run's exclusions when it reaches
    DONE, so a run that is killed or abandoned mid-flight bans its nodes
    from every later run forever. fish-lean-longtest -- a lean smoke run
    over a deliberately tiny shard subset, where having "no shards left to
    assign" was the expected outcome -- is how node002, node005, node006 and
    node011 left the fish tier permanently.

    Fails closed: a reason this module cannot attribute to a specific run is
    never reclaimed.
    """
    live = set(live_run_ids)
    stale: dict[str, str] = {}
    for node, reason in excluded_nodes.items():
        for pattern in _RUN_SCOPED_REASONS:
            match = pattern.match(reason)
            if match and match.group("run_id") not in live:
                stale[node] = reason
                break
    return stale
