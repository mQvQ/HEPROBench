#!/usr/bin/env python3
"""Download and verify the public real-data CRC-CODEX reviewer demo."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO_ID = "u3011706/HEPROBench-CRC-CODEX-review-demo"
REVISION = "442b41a1c7794993508558c899361b98d52e2879"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(root: Path) -> int:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"Checksum mismatch for {relative}: {observed} != {expected}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reviewer_data" / "crc_codex",
    )
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is required; install requirements-full.txt or run in the documented conda environment"
        ) from exc

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=output,
        token=False,
        etag_timeout=30,
    )
    checked = _verify(output)
    print(f"Verified {checked} files at {output}")
    print(f"Hugging Face revision: {REVISION}")


if __name__ == "__main__":
    main()
