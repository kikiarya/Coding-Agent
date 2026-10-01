"""Export the current source files without Git metadata or local run artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {".github", "configs", "evals", "examples", "schemas", "scripts", "src", "tests", "web"}
ROOT_FILES = {"README.md", "LICENSE", ".gitignore", "pyproject.toml", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md"}
EXTRA_FILES = {
    "assets/architecture.mmd", "assets/architecture.svg", "assets/architecture.png",
    "scripts/render-architecture.ps1", "scripts/export-public-source.py",
    "docs/console-preview.svg",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT,
    ).decode("utf-8").split("\0")
    selected = set()
    for name in tracked:
        if not name:
            continue
        path = PurePosixPath(name)
        if name in ROOT_FILES or path.parts[0] in SOURCE_DIRS:
            selected.add(name)
    selected.update(name for name in EXTRA_FILES if (ROOT / name).is_file())
    selected.update(name for name in ROOT_FILES if (ROOT / name).is_file())
    for name in selected:
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"Expected a regular source file: {name}")
        if ROOT not in path.resolve().parents:
            raise SystemExit(f"Source path escapes repository: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(selected):
            archive.write(ROOT / name, f"coding-agent-policy-optimization/{name}")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise SystemExit("Archive integrity check failed")
    print(f"Exported {len(selected)} files to {output}")
    print(f"SHA256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
