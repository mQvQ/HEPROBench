from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from heprobench.preprocess import STAGES, _legacy_normalize, run_preprocessing


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = ROOT / "configs" / "preprocessing" / "crc_codex.json"


class PreprocessingTests(unittest.TestCase):
    def test_crc_codex_dry_run_lists_complete_pipeline(self) -> None:
        result = run_preprocessing(REFERENCE_CONFIG, dry_run=True)
        self.assertEqual(result["dataset"], "CRC-CODEX")
        self.assertEqual(result["stages"], list(STAGES))
        self.assertTrue(result["dry_run"])

    def test_legacy_normalization_does_not_divide_by_log2(self) -> None:
        target = np.asarray([[[0.0], [100.0]]], dtype=np.float32)
        legacy = _legacy_normalize(target, np.asarray([100.0]), divide_by_log2=False)
        corrected = _legacy_normalize(target, np.asarray([100.0]), divide_by_log2=True)
        self.assertEqual(int(legacy[0, 0, 0]), 0)
        self.assertEqual(int(legacy[0, 1, 0]), int(np.log(2.0) * 255.0))
        self.assertEqual(int(corrected[0, 1, 0]), 255)

    def test_split_keeps_patient_slides_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            manifest = data_root / "manifests" / "slides.csv"
            manifest.parent.mkdir(parents=True)
            rows = []
            for patient in range(10):
                for slide in range(2):
                    rows.append(
                        {
                            "slide_id": f"p{patient}_s{slide}",
                            "patient_id": f"p{patient}",
                            "he_path": f"he/{patient}_{slide}.tif",
                            "target_path": f"codex/{patient}_{slide}.tif",
                            "split": "",
                        }
                    )
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            channels = root / "channels.json"
            channels.write_text('["DRAQ5"]\n', encoding="utf-8")
            config = {
                "schema_version": "heprobench_preprocessing_v1",
                "dataset": {
                    "name": "test",
                    "root_dir": str(data_root),
                    "slide_manifest": "manifests/slides.csv",
                    "channel_names_file": str(channels),
                    "columns": {
                        "slide_id": "slide_id",
                        "patient_id": "patient_id",
                        "he_path": "he_path",
                        "target_path": "target_path",
                    },
                },
                "split": {
                    "group_column": "patient_id",
                    "split_column": "split",
                    "reuse_existing": True,
                    "ratios": {"train": 0.7, "valid": 0.1, "test": 0.2},
                    "seed": 42,
                },
                "output": {"root_dir": str(root / "output")},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = run_preprocessing(config_path, stages=["split"])
            split_path = root / "output" / "manifests" / "slides_with_split.csv"
            with split_path.open(encoding="utf-8", newline="") as handle:
                split_rows = list(csv.DictReader(handle))
            assignments: dict[str, set[str]] = {}
            for row in split_rows:
                assignments.setdefault(row["patient_id"], set()).add(row["split"])
            self.assertTrue(all(len(values) == 1 for values in assignments.values()))
            self.assertEqual({row["split"] for row in split_rows}, {"train", "valid", "test"})
            self.assertEqual(result["stages_completed_this_run"], ["split"])

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("PIL", "cv2", "tdigest", "sklearn")),
        "preprocessing extras are not installed",
    )
    def test_image_qc_cell_and_gating_stages_connect(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            channels = root / "channels.json"
            channels.write_text('["Marker", "DRAQ5"]\n', encoding="utf-8")
            config = {
                "schema_version": "heprobench_preprocessing_v1",
                "dataset": {
                    "name": "test",
                    "root_dir": str(root / "raw"),
                    "slide_manifest": "unused.csv",
                    "channel_names_file": str(channels),
                },
                "registration_qc": {"require_manual_acceptance": True},
                "tiling": {
                    "patch_size": 16,
                    "stride": 16,
                    "raw_channel_count": 2,
                    "target_channel_axis": "auto",
                    "retained_channel_indices": [0, 1],
                },
                "normalization": {
                    "fit_split": "train",
                    "percentile": 99.9,
                    "samples_per_patch": 256,
                    "divide_by_log2": False,
                },
                "patch_qc": {
                    "nuclear_channel": "DRAQ5",
                    "he_inverse_channel": 0,
                    "nuclear_std_min": -1,
                    "robust_nmi_min": -1,
                    "gaussian_sigma": 1.5,
                    "robust_nmi_bins": 16,
                    "diagnostic_nmi_bins": 64,
                    "min_foreground_pixels": 1,
                },
                "gating": {
                    "positive_only_fit": True,
                    "posterior_threshold": 0.5,
                    "max_cells_per_marker": 1000,
                    "min_cells_to_fit": 2,
                    "seed": 42,
                    "separation_min": 0,
                    "positive_rate_min": 0,
                    "positive_rate_max": 1,
                },
                "output": {"root_dir": str(output)},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            qc_rows = []
            for index, split in enumerate(("train", "valid", "test")):
                slide = f"slide_{index}"
                yy, xx = np.indices((32, 32))
                nuclear = np.uint16((xx + yy + index * 7) % 31 + 1)
                marker = np.where(xx < 16, 20, 200).astype(np.uint16)
                target = np.stack([marker, nuclear], axis=-1)
                he = np.stack([255 - np.uint8(nuclear * 8), np.full_like(nuclear, 180), np.full_like(nuclear, 210)], axis=-1)
                he_path, target_path = root / f"{slide}.png", root / f"{slide}.npy"
                Image.fromarray(np.uint8(he)).save(he_path)
                np.save(target_path, target)
                qc_rows.append(
                    {
                        "slide_id": slide,
                        "patient_id": f"patient_{index}",
                        "split": split,
                        "registered_he_path": str(he_path),
                        "registered_target_path": str(target_path),
                        "decision": "accepted",
                        "notes": "fixture",
                    }
                )
            qc_path = output / "registration_qc" / "registration_qc.csv"
            qc_path.parent.mkdir(parents=True)
            with qc_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(qc_rows[0]))
                writer.writeheader()
                writer.writerows(qc_rows)

            run_preprocessing(config_path, stages=["tile", "normalize", "patch-qc"])
            normalized_path = output / "manifests" / "patches_normalized.csv"
            with normalized_path.open(encoding="utf-8", newline="") as handle:
                patch_rows = list(csv.DictReader(handle))
            masked_rows = []
            for patch_index, row in enumerate(patch_rows):
                mask = np.zeros((16, 16), dtype=np.int32)
                local = patch_index % 4
                base = local * 4 + 1
                mask[:8, :8], mask[:8, 8:], mask[8:, :8], mask[8:, 8:] = base, base + 1, base + 2, base + 3
                mask_path = output / "patches" / "masks" / row["slide_name"] / f"{patch_index}.npy"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(mask_path, mask)
                masked_rows.append({**row, "mask_path": str(mask_path.relative_to(output))})
            masked_manifest = output / "manifests" / "patches_with_masks.csv"
            with masked_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(masked_rows[0]))
                writer.writeheader()
                writer.writerows(masked_rows)

            run_preprocessing(config_path, stages=["cell-extract", "gate"])
            self.assertTrue((output / "cell_annotations.csv").is_file())
            self.assertTrue((output / "gmm_marker_qc.csv").is_file())
            with (output / "cell_annotations.csv").open(encoding="utf-8", newline="") as handle:
                cells = list(csv.DictReader(handle))
            self.assertEqual(len(cells), 48)
            self.assertIn("Marker_pos", cells[0])


if __name__ == "__main__":
    unittest.main()
