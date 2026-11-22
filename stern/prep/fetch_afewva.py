#!/usr/bin/env python
"""Extract downloaded AFEW-VA archives into the STERN data root.

AFEW-VA is EULA-gated (ibug, Imperial College): request access at
https://ibug.doc.ic.ac.uk/resources/afew-va-database/ and download the 12 zip
archives manually -- there is no anonymous download URL, so this script cannot
fetch them for you. It only unpacks the zips you already downloaded.

Defaults to `.data/afewva` (see `.gitignore`'s "local bulk dataset downloads"
entry) for local work. On the login-node/SallesInfo cluster, override with
`AFEWVA_ROOT=/Data/$USER/afewva` -- bulk data belongs on per-node `/Data`
there, never the shared NFS mount.

Usage (local):
    uv run python stern/prep/fetch_afewva.py --zips ~/Downloads/AFEW-VA/*.zip

Usage (cluster):
    AFEWVA_ROOT=/Data/$USER/afewva \\
      uv run python stern/prep/fetch_afewva.py --zips ~/Downloads/AFEW-VA/*.zip

Then build the manifest (PYTHONPATH=. is needed for build_afewva_manifest.py's
`stern.data.manifest` import; this fetch script doesn't need it):
    PYTHONPATH=. uv run python stern/prep/build_afewva_manifest.py --afewva-root $AFEWVA_ROOT
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


def extract(zips: list[Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for archive in zips:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        print(f"extracted {archive} -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(os.environ.get("AFEWVA_ROOT", ".data/afewva")),
    )
    args = parser.parse_args()

    extract(args.zips, args.dest)
    print(
        "next: PYTHONPATH=. uv run python stern/prep/build_afewva_manifest.py "
        f"--afewva-root {args.dest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
