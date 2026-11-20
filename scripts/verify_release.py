"""Verify saved report bytes; optionally require current source snapshots."""

import argparse
import hashlib
import json
from pathlib import Path

STUDIES = [
    ".",
    "robustness",
    "sequential",
    "adjustment",
    "fieller",
    "ratio_sensitivity",
]


def verify(reports, source=None):
    reports = Path(reports)
    count = 0
    for study in STUDIES:
        folder = reports / study
        path = folder / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("artifact_sha256"):
            raise ValueError(f"No artifact checksums: {path}")
        for name, expected in data["artifact_sha256"].items():
            artifact = (folder / name).resolve()
            if not artifact.is_relative_to(folder.resolve()):
                raise ValueError(f"Artifact escapes report directory: {name}")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {artifact}")
            count += 1
        if source is not None:
            source = Path(source)
            actual = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.glob("*.py")
            }
            if actual != data.get("source_sha256"):
                raise ValueError(f"Source snapshot mismatch: {path}")
    return count


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=root / "reports")
    parser.add_argument("--current-source", action="store_true")
    args = parser.parse_args()
    try:
        count = verify(
            args.reports, root / "experimentlab" if args.current_source else None
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Verified {count} artifacts across six study manifests.")


if __name__ == "__main__":
    main()
