from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from heprobench.preprocess import STAGES, _legacy_normalize, load_preprocessing_config, run_preprocessing


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = ROOT / "configs" / "preprocessing" / "crc_codex.json"
DATASET_CONFIGS = {
    "crc_codex.json": ("CRC-CODEX", 58, "verified"),
    "aml_codex.json": ("AML-CODEX", 54, "verified"),
    "mt_codex.json": ("MT-CODEX", 60, "verified"),
    "spatch_codex.json": ("SPATCH-CODEX", 17, "pending_source_audit"),
    "nsclc_imc.json": ("NSCLC-IMC", 28, "verified"),
    "hemit.json": ("HEMIT", 3, "verified"),
    "crc_mihc_p1.json": ("CRC-mIHC-P1", 7, "partially_verified"),
    "crc_mihc_p2.json": ("CRC-mIHC-P2", 7, "partially_verified"),
    "stad_mihc.json": ("STAD-mIHC", 11, "pending_source_audit"),
}


class PreprocessingTests(unittest.TestCase):
    def test_all_dataset_configs_are_valid_and_declare_access(self) -> None:
        for filename, (name, channels, status) in DATASET_CONFIGS.items():
            with self.subTest(filename=filename):
                config_path = ROOT / "configs" / "preprocessing" / filename
                result = run_preprocessing(config_path, dry_run=True)
                config = load_preprocessing_config(config_path)["config"]
                self.assertEqual(result["dataset"], name)
                self.assertEqual(result["channel_count"], channels)
                self.assertEqual(result["reproduction_status"], status)
                self.assertNotEqual(result["access_status"], "unspecified")
                self.assertFalse(config["dataset"]["access"]["data_in_repository"])
                self.assertNotIn("legacy_id", config["dataset"])

    def test_hemit_raw_channels_are_reordered_to_canonical_panel(self) -> None:
        ctx = load_preprocessing_config(ROOT / "configs" / "preprocessing" / "hemit.json")
        tiling = ctx["config"]["tiling"]
        self.assertEqual(tiling["raw_channel_order"], ["PanCK", "CD3", "DAPI"])
        self.assertEqual(tiling["retained_channel_indices"], [2, 0, 1])

    def test_committed_manifest_templates_are_fictional_and_path_neutral(self) -> None:
        template_dir = ROOT / "configs" / "preprocessing" / "templates"
        for path in template_dir.glob("*.csv"):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("sample_001", text)
                self.assertIn("group_001", text)
                self.assertNotIn("/home/", text)
                self.assertNotIn("/data", text)
                self.assertNotIn("\\Users\\", text)

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

    @unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "Pillow is not installed")
    def test_pre_registered_wsi_is_partitioned_into_fovs(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            (data_root / "manifests").mkdir(parents=True)
            he_path = data_root / "he.png"
            target_path = data_root / "target.npy"
            Image.fromarray(np.zeros((8, 16, 3), dtype=np.uint8)).save(he_path)
            np.save(target_path, np.ones((8, 16, 1), dtype=np.uint16))
            manifest = data_root / "manifests" / "slides.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sample_id", "slide_id", "fov_id", "group_id", "he_path", "target_path", "split"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "sample_001",
                        "slide_id": "slide_001",
                        "fov_id": "sample_001",
                        "group_id": "group_001",
                        "he_path": "he.png",
                        "target_path": "target.npy",
                        "split": "train",
                    }
                )
            channels = root / "channels.json"
            channels.write_text('["DAPI"]\n', encoding="utf-8")
            config = {
                "schema_version": "heprobench_preprocessing_v1",
                "dataset": {
                    "name": "fixture",
                    "root_dir": str(data_root),
                    "slide_manifest": "manifests/slides.csv",
                    "channel_names_file": str(channels),
                    "columns": {
                        "sample_id": "sample_id",
                        "slide_id": "slide_id",
                        "fov_id": "fov_id",
                        "he_path": "he_path",
                        "target_path": "target_path",
                    },
                },
                "split": {"group_column": "group_id", "split_column": "split", "reuse_existing": True},
                "registration": {"method": "pre_registered"},
                "registration_qc": {"enabled": False},
                "tiling": {
                    "patch_size": 4,
                    "stride": 4,
                    "fov_size_px": 8,
                    "raw_channel_count": 1,
                    "target_channel_axis": "auto",
                    "retained_channel_indices": [0],
                },
                "output": {"root_dir": str(root / "output")},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            run_preprocessing(
                config_path,
                stages=["split", "register", "registration-qc", "tile"],
            )
            with (root / "output" / "manifests" / "patches_raw.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({row["fov_name"] for row in rows}), 2)
            self.assertEqual(len({(row["slide_name"], row["row"], row["col"]) for row in rows}), 8)

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
