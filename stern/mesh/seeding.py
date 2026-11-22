"""Verify that a pool's on-disk shard payload matches the golden reference.

Written after the 2026-08-20 factorial incident, in which shard *attribution*
and shard *reality* disagreed for two weeks without anything noticing: the
shard map claimed node069 held 89 shards; node069 held 0. Assignment was computed
against one source of truth and resolution against another, so workers were
assigned shards that did not exist, trained on zero samples, and reported it
as a warning.

Everything here is pure. The SSH scan that feeds it lives in
`stern/mesh/validate_seeding.py`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeReport:
    node: str
    missing: tuple[str, ...]
    wrong_size: tuple[str, ...]
    unexpected: tuple[str, ...]
    val_missing: tuple[str, ...]
    manifest_missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        # `unexpected` is deliberately excluded: leftover files from an older
        # seeding are noise, not a failure. Everything else breaks training.
        return not (self.wrong_size or self.val_missing or self.manifest_missing)


@dataclass(frozen=True)
class ValidationReport:
    pool: str
    nodes: tuple[NodeReport, ...]
    coverage: float
    missing_pool_wide: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            bool(self.nodes)
            and not self.missing_pool_wide
            and all(n.ok for n in self.nodes)
        )


def validate(
    pool: str,
    manifest: dict[str, int],
    val_manifest: dict[str, int],
    scan: dict[str, dict[str, int]],
    val_scan: dict[str, dict[str, int]],
    has_manifest: dict[str, tuple[bool, bool]],
) -> ValidationReport:
    """Compare a scanned pool against the golden manifests.

    `scan` / `val_scan` map node -> {filename: byte size}. `has_manifest` maps
    node -> (train manifest.json present, val/manifest.json present).

    A shard present at the wrong size does NOT count toward coverage: a
    half-transferred shard resolves locally and then fails mid-epoch, which is
    strictly worse than an absent one.
    """
    reports: list[NodeReport] = []
    good: set[str] = set()

    for node in sorted(scan):
        found = scan[node]
        missing = tuple(sorted(n for n in manifest if n not in found))
        wrong_size = tuple(
            sorted(n for n, size in manifest.items() if n in found and found[n] != size)
        )
        unexpected = tuple(sorted(n for n in found if n not in manifest))
        good.update(n for n, size in manifest.items() if found.get(n) == size)

        val_found = val_scan.get(node, {})
        val_missing = tuple(
            sorted(n for n, size in val_manifest.items() if val_found.get(n) != size)
        )

        train_ok, val_ok = has_manifest.get(node, (False, False))
        manifest_missing = tuple(
            name
            for name, present in (
                ("manifest.json", train_ok),
                ("val/manifest.json", val_ok),
            )
            if not present
        )

        reports.append(
            NodeReport(
                node=node,
                missing=missing,
                wrong_size=wrong_size,
                unexpected=unexpected,
                val_missing=val_missing,
                manifest_missing=manifest_missing,
            )
        )

    coverage = len(good) / len(manifest) if manifest else 0.0
    missing_pool_wide = tuple(sorted(n for n in manifest if n not in good))
    return ValidationReport(
        pool=pool,
        nodes=tuple(reports),
        coverage=coverage,
        missing_pool_wide=missing_pool_wide,
    )


def holdings_from_scan(scan: dict[str, dict[str, int]]) -> dict[str, list[str]]:
    """Drop sizes, keeping a deterministic node -> [shard, ...] mapping.

    Plain dicts and lists, not tuples: this is serialized to
    verified_holdings.json and must survive a JSON round-trip unchanged.
    """
    return {node: sorted(scan[node]) for node in sorted(scan)}
