#!/usr/bin/env python3
"""Create the deidentified real CRC-CODEX reviewer-demo bundle.

The source directory is treated as read-only.  The output contains only a
small marker-balanced subset of QC-passing patches and never records source
FOV names, patient mappings, clinical outcomes, or local filesystem paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image


SOURCE_DOI = "10.17632/mpjzbtfgfr.1"
SOURCE_TITLE = (
    "Coordinated cellular neighborhoods orchestrate antitumoral immunity "
    "at the colorectal cancer invasive front"
)
SOURCE_CREATOR = "Christian Schürch"
SOURCE_LICENSE = "CC BY 4.0"
SOURCE_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SOURCE_RETRIEVED_ON = "2026-09-05"
SPLITS = ("train", "valid", "test")
CLASSIFICATION_MARKERS = ("CD3", "CD20", "Cytokeratin")
CHANNELS = (
    ("DAPI", "DRAQ5", 57, True),
    ("CD3", "CD3", 45, False),
    ("CD20", "CD20", 20, False),
    ("PanCK", "Cytokeratin", 25, True),
)
METADATA_FIELDS = ("slide_name", "row", "col", "split", "image_path", "target_path", "mask_path")
PATCH_RE = re.compile(r"_codex_patch_(\d+)_(\d+)$")


@dataclass(frozen=True)
class Candidate:
    split: str
    source_slide: str
    image_rel: str
    target_rel: str
    stem: str
    row: int
    col: int
    cell_count: int
    positive_counts: tuple[int, ...]

    @property
    def rank(self) -> tuple[int, int, int, str]:
        return (min(self.positive_counts), sum(self.positive_counts), self.cell_count, self.stem)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Read-only preprocessed CRC-CODEX root")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, empty reviewer-bundle directory")
    parser.add_argument("--slides-per-split", type=int, default=2)
    parser.add_argument("--patches-per-slide", type=int, default=2)
    return parser.parse_args()


def _decode(values: Iterable[Any]) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _safe_source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Source metadata path escapes source directory: {relative}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _mask_path(root: Path, target_rel: str) -> Path:
    mask_rel = target_rel.replace("if_image/", "slide-seg-deepcell-mask-mesmer/").replace(
        "_codex_patch_", "_mask_patch_"
    )
    return _safe_source_path(root, mask_rel)


def _read_metadata(root: Path, split: str) -> list[dict[str, str]]:
    path = root / f"{split}_filter_dapi_std_11_inv_red_nmi_003_patch_meta.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "target_path", "slide_name"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected metadata schema: {path}")
    return rows


def _candidate_stats(root: Path, split: str) -> list[Candidate]:
    by_slide: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in _read_metadata(root, split):
        stem = Path(row["target_path"]).stem
        by_slide[row["slide_name"]][stem] = row

    candidates: list[Candidate] = []
    for source_slide, metadata_by_stem in by_slide.items():
        h5_path = root / "fov-stage0-h5ad" / source_slide / "stage0_fov_gmm.h5ad"
        if not h5_path.is_file():
            raise FileNotFoundError(h5_path)
        with h5py.File(h5_path, "r") as handle:
            categories = _decode(handle["obs/patch_id_from_centroid/categories"][:])
            codes = handle["obs/patch_id_from_centroid/codes"][:]
            is_eval = handle["obs/is_eval_cell"][:].astype(bool, copy=False)
            marker_positive = {
                marker: handle[f"obs/{marker}_pos"][:].astype(bool, copy=False)
                for marker in CLASSIFICATION_MARKERS
            }
            for code, stem in enumerate(categories):
                metadata = metadata_by_stem.get(stem)
                if metadata is None:
                    continue
                selected = (codes == code) & is_eval
                cell_count = int(selected.sum())
                positive_counts = tuple(int(marker_positive[marker][selected].sum()) for marker in CLASSIFICATION_MARKERS)
                negative_counts = tuple(cell_count - count for count in positive_counts)
                if cell_count == 0 or min(positive_counts) < 1 or min(negative_counts) < 1:
                    continue
                match = PATCH_RE.search(stem)
                if match is None:
                    raise ValueError(f"Cannot parse patch coordinates: {stem}")
                candidates.append(
                    Candidate(
                        split=split,
                        source_slide=source_slide,
                        image_rel=metadata["image_path"],
                        target_rel=metadata["target_path"],
                        stem=stem,
                        row=int(match.group(1)),
                        col=int(match.group(2)),
                        cell_count=cell_count,
                        positive_counts=positive_counts,
                    )
                )
    return candidates


def _select_candidates(
    candidates: list[Candidate],
    *,
    slides_per_split: int,
    patches_per_slide: int,
) -> list[Candidate]:
    by_slide: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_slide[candidate.source_slide].append(candidate)

    ranked_slides: list[tuple[tuple[int, int, int, str], str, list[Candidate]]] = []
    for source_slide, slide_candidates in by_slide.items():
        chosen = sorted(slide_candidates, key=lambda item: item.rank, reverse=True)[:patches_per_slide]
        if len(chosen) != patches_per_slide:
            continue
        slide_rank = (
            sum(min(item.positive_counts) for item in chosen),
            sum(sum(item.positive_counts) for item in chosen),
            sum(item.cell_count for item in chosen),
            source_slide,
        )
        ranked_slides.append((slide_rank, source_slide, chosen))
    ranked_slides.sort(reverse=True)
    selected_slides = ranked_slides[:slides_per_split]
    if len(selected_slides) != slides_per_split:
        raise ValueError(
            f"Only {len(selected_slides)} slides satisfy selection requirements; requested {slides_per_split}"
        )
    return [candidate for _, _, chosen in selected_slides for candidate in chosen]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_card(*, patches_per_split: int, slides_per_split: int, cell_counts: dict[str, int]) -> str:
    return f"""---
license: cc-by-4.0
task_categories:
- image-to-image
tags:
- histopathology
- codex
- virtual-staining
- reviewer-demo
pretty_name: HEPROBench CRC-CODEX Reviewer Demo
size_categories:
- n<1K
---

# HEPROBench CRC-CODEX reviewer demo

This is a small **real-data** software-verification subset for HEPROBench. It
contains paired, registered H&E and CRC-CODEX patches derived from Schürch et
al., *Coordinated cellular neighborhoods orchestrate antitumoral immunity at
the colorectal cancer invasive front*, created by {SOURCE_CREATOR}, Mendeley Data
[{SOURCE_DOI}](https://doi.org/{SOURCE_DOI}), licensed under
[{SOURCE_LICENSE}]({SOURCE_LICENSE_URL}).

The bundle has {patches_per_split} patches from each of {slides_per_split}
anonymized FOVs per split ({patches_per_split * slides_per_split} train,
{patches_per_split * slides_per_split} validation, and
{patches_per_split * slides_per_split} test patches). Cell counts in the
included masks are train={cell_counts['train']}, valid={cell_counts['valid']},
and test={cell_counts['test']} after aggregation by anonymous FOV and cell ID.

## Contents

- `images/`: 256x256 RGB H&E JPEGs, re-encoded without EXIF metadata.
- `targets/`: aligned 256x256x4 `uint8` arrays (`DAPI`, `CD3`, `CD20`, `PanCK`).
- `masks/`: 256x256 non-negative integer Mesmer cell-ID masks; zero is background.
- `metadata.csv` and `splits/*.csv`: the HEPROBench patch contract.
- `cell_annotations.csv`: continuous cell means and GMM-derived positive labels.
- `channel_stats.json`, `gmm_gates.json`, `provenance.json`, and `SHA256SUMS`:
  normalization, label, provenance, and integrity records.

The four reviewer channels are an exact subset of the normalized 58-channel
arrays: DRAQ5 is exposed as DAPI, CD3 as CD3, CD20 as CD20, and Cytokeratin as
PanCK. Values already lie in the 8-bit range, so selecting the four channels
and storing them as `uint8` does not rescale or otherwise transform intensity.

The source landing page and its CC BY 4.0 license link were publicly accessible
on {SOURCE_RETRIEVED_ON}. The DataCite record also contained a legacy
`embargoedAccess` entry; current public access and the explicit license were
therefore checked directly before this release.

## Scope and privacy

This subset is selected for marker/class diversity and is intended only to
verify train -> validation/test inference -> image/per-slide/cell evaluation.
It is not representative of the full cohort and must not be used to reproduce
the paper's scientific performance numbers.

No patient identifiers, original TMA/core names, clinical outcomes, source
paths, or source metadata tables are included. FOV and file names are newly
assigned per split. JPEG metadata are stripped during export. The images are
derived from human biospecimens and should still be handled according to the
source dataset's terms and the CC BY 4.0 attribution requirement.
"""


def _prepare(args: argparse.Namespace) -> Path:
    source = args.source_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if args.slides_per_split < 1 or args.patches_per_slide < 1:
        raise ValueError("slides-per-split and patches-per-slide must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("images", "targets", "masks", "splits"):
        (output / name).mkdir()

    gmm_path = source / "gmm_params.json"
    with gmm_path.open("r", encoding="utf-8") as handle:
        source_gmm = json.load(handle)
    gates = {
        public: {
            "source_marker": source_name,
            "threshold_intersection": float(source_gmm[source_name]["threshold_intersection"]),
            "source_channel_index": index,
        }
        for public, source_name, index, _ in CHANNELS
    }

    metadata_rows: list[dict[str, Any]] = []
    selected_summary: dict[str, dict[str, int]] = {}
    target_arrays: dict[tuple[str, int, int], np.ndarray] = {}
    mask_arrays: dict[tuple[str, int, int], np.ndarray] = {}
    rgb_moments = [0, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)]
    target_moments = [0, np.zeros(len(CHANNELS), dtype=np.float64), np.zeros(len(CHANNELS), dtype=np.float64)]

    for split in SPLITS:
        selected = _select_candidates(
            _candidate_stats(source, split),
            slides_per_split=args.slides_per_split,
            patches_per_slide=args.patches_per_slide,
        )
        source_slides: list[str] = []
        for candidate in selected:
            if candidate.source_slide not in source_slides:
                source_slides.append(candidate.source_slide)
        anonymous_names = {
            source_slide: f"crc_codex_{split}_{index:03d}"
            for index, source_slide in enumerate(source_slides, start=1)
        }
        selected_summary[split] = {
            "slides": len(source_slides),
            "patches": len(selected),
        }

        for candidate in selected:
            slide_name = anonymous_names[candidate.source_slide]
            stem = f"{slide_name}_patch_{candidate.row:03d}_{candidate.col:03d}"
            image_path = _safe_source_path(source, candidate.image_rel)
            target_path = _safe_source_path(source, candidate.target_rel)
            mask_path = _mask_path(source, candidate.target_rel)

            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            target_full = np.load(target_path, allow_pickle=False)
            mask = np.load(mask_path, allow_pickle=False)
            if rgb.shape != (256, 256, 3):
                raise ValueError(f"Unexpected H&E shape: {image_path}: {rgb.shape}")
            if target_full.ndim != 3 or target_full.shape[:2] != (256, 256) or target_full.shape[2] < 58:
                raise ValueError(f"Unexpected target shape: {target_path}: {target_full.shape}")
            if mask.shape != (256, 256) or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
                raise ValueError(f"Unexpected cell mask: {mask_path}")
            target = target_full[..., [channel[2] for channel in CHANNELS]]
            if np.min(target) < 0 or np.max(target) > 255:
                raise ValueError(f"Selected target channels are outside uint8 range: {target_path}")
            target = target.astype(np.uint8, copy=False)
            mask = mask.astype(np.int32, copy=False)

            image_rel = f"images/{stem}.jpg"
            target_rel = f"targets/{stem}.npy"
            mask_rel = f"masks/{stem}.npy"
            Image.fromarray(rgb).save(
                output / image_rel,
                format="JPEG",
                quality=95,
                subsampling=0,
                exif=b"",
            )
            np.save(output / target_rel, target, allow_pickle=False)
            np.save(output / mask_rel, mask, allow_pickle=False)

            key = (slide_name, candidate.row, candidate.col)
            target_arrays[key] = target
            mask_arrays[key] = mask
            metadata_rows.append(
                {
                    "slide_name": slide_name,
                    "row": candidate.row,
                    "col": candidate.col,
                    "split": split,
                    "image_path": image_rel,
                    "target_path": target_rel,
                    "mask_path": mask_rel,
                }
            )
            if split == "train":
                pixels = rgb.reshape(-1, 3).astype(np.float64)
                rgb_moments[0] += len(pixels)
                rgb_moments[1] += pixels.sum(axis=0)
                rgb_moments[2] += np.square(pixels).sum(axis=0)
                values = target.reshape(-1, len(CHANNELS)).astype(np.float64)
                target_moments[0] += len(values)
                target_moments[1] += values.sum(axis=0)
                target_moments[2] += np.square(values).sum(axis=0)

    metadata_rows.sort(key=lambda row: (SPLITS.index(str(row["split"])), row["slide_name"], row["row"], row["col"]))
    _write_csv(output / "metadata.csv", metadata_rows, METADATA_FIELDS)
    for split in SPLITS:
        _write_csv(
            output / "splits" / f"{split}.csv",
            [row for row in metadata_rows if row["split"] == split],
            METADATA_FIELDS,
        )

    accumulators: dict[tuple[str, int, str], tuple[np.ndarray, int]] = {}
    for row in metadata_rows:
        key = (str(row["slide_name"]), int(row["row"]), int(row["col"]))
        target = target_arrays[key]
        mask = mask_arrays[key]
        ids = mask.reshape(-1).astype(np.int64, copy=False)
        pixels = target.reshape(-1, len(CHANNELS)).astype(np.float64, copy=False)
        unique, inverse = np.unique(ids, return_inverse=True)
        sums = np.zeros((len(unique), len(CHANNELS)), dtype=np.float64)
        np.add.at(sums, inverse, pixels)
        counts = np.bincount(inverse, minlength=len(unique))
        for index, cell_id in enumerate(unique.tolist()):
            if cell_id == 0:
                continue
            cell_key = (str(row["slide_name"]), int(cell_id), str(row["split"]))
            previous = accumulators.get(cell_key)
            if previous is None:
                accumulators[cell_key] = (sums[index], int(counts[index]))
            else:
                accumulators[cell_key] = (previous[0] + sums[index], previous[1] + int(counts[index]))

    annotation_rows: list[dict[str, Any]] = []
    for (slide_name, cell_id, split), (sums, count) in sorted(accumulators.items()):
        means = sums / count
        annotation: dict[str, Any] = {
            "slide_name": slide_name,
            "global_cell_id": cell_id,
            "split": split,
        }
        for index, (public, _, _, _) in enumerate(CHANNELS):
            value = float(means[index])
            annotation[public] = value
            annotation[f"{public}_pos"] = int(value >= gates[public]["threshold_intersection"])
        annotation_rows.append(annotation)
    annotation_fields = ["slide_name", "global_cell_id", "split"]
    for public, _, _, _ in CHANNELS:
        annotation_fields.extend([public, f"{public}_pos"])
    _write_csv(output / "cell_annotations.csv", annotation_rows, annotation_fields)

    class_counts: dict[str, dict[str, dict[str, int]]] = {}
    for split in SPLITS:
        rows = [row for row in annotation_rows if row["split"] == split]
        class_counts[split] = {}
        for public in ("CD3", "CD20", "PanCK"):
            positive = sum(int(row[f"{public}_pos"]) for row in rows)
            negative = len(rows) - positive
            if positive == 0 or negative == 0:
                raise ValueError(f"{split}/{public} does not contain both cell classes")
            class_counts[split][public] = {"positive": positive, "negative": negative}

    target_count, target_sum, target_square_sum = target_moments
    rgb_count, rgb_sum, rgb_square_sum = rgb_moments
    target_mean = target_sum / target_count
    target_std = np.sqrt(np.maximum(target_square_sum / target_count - np.square(target_mean), 0.0))
    rgb_mean = rgb_sum / rgb_count
    rgb_std = np.sqrt(np.maximum(rgb_square_sum / rgb_count - np.square(rgb_mean), 0.0))
    channel_stats: dict[str, Any] = {}
    for index, (public, _, _, structural) in enumerate(CHANNELS):
        channel_stats[public] = {
            "mean": float(target_mean[index]),
            "std": float(target_std[index]),
            "channel_idx": index,
            "idx_channel": index,
            "is_structural": structural,
        }
    channel_stats["RGB"] = {"mean": rgb_mean.tolist(), "std": rgb_std.tolist()}
    (output / "channel_names.json").write_text(
        json.dumps([channel[0] for channel in CHANNELS], indent=2) + "\n", encoding="utf-8"
    )
    (output / "channel_stats.json").write_text(json.dumps(channel_stats, indent=2) + "\n", encoding="utf-8")
    (output / "gmm_gates.json").write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")

    cell_counts = {split: sum(row["split"] == split for row in annotation_rows) for split in SPLITS}
    provenance = {
        "schema": "heprobench_crc_codex_reviewer_demo_v1",
        "source": {
            "title": SOURCE_TITLE,
            "creator": SOURCE_CREATOR,
            "doi": SOURCE_DOI,
            "license": SOURCE_LICENSE,
            "license_url": SOURCE_LICENSE_URL,
            "landing_page": "https://data.mendeley.com/datasets/mpjzbtfgfr/1",
            "observed_access": "public",
            "retrieved_on": SOURCE_RETRIEVED_ON,
            "gmm_parameters_sha256": _sha256(gmm_path),
            "metadata_sha256": {
                split: _sha256(source / f"{split}_filter_dapi_std_11_inv_red_nmi_003_patch_meta.csv")
                for split in SPLITS
            },
        },
        "selection": {
            "strategy": "deterministic marker-balanced ranking among QC-passing patches",
            "classification_markers": list(CLASSIFICATION_MARKERS),
            "slides_per_split": args.slides_per_split,
            "patches_per_slide": args.patches_per_slide,
            "summary": selected_summary,
            "cell_class_counts": class_counts,
        },
        "channels": gates,
        "privacy": {
            "anonymous_fov_names": True,
            "source_fov_names_in_bundle": False,
            "patient_identifiers_in_bundle": False,
            "clinical_outcomes_in_bundle": False,
            "local_paths_in_bundle": False,
            "jpeg_exif_removed": True,
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        _dataset_card(
            patches_per_split=args.patches_per_slide,
            slides_per_split=args.slides_per_split,
            cell_counts=cell_counts,
        ),
        encoding="utf-8",
    )

    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    checksum_lines = [f"{_sha256(path)}  {path.relative_to(output).as_posix()}" for path in files]
    (output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    args = _parse_args()
    output = _prepare(args)
    print(json.dumps({"status": "prepared", "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
