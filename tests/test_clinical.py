from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from heprobench.clinical_metrics import harrell_c_index
from heprobench.clinical_pipeline import run_clinical_pipeline


class ClinicalPipelineTests(unittest.TestCase):
    def _config(self, root: Path, *, task: str = "survival") -> Path:
        payload = {
            "schema_version": "heprobench_clinical_v1",
            "clinical": {
                "task": task,
                "class_names": ["negative", "positive"],
                "columns": {},
            },
            "data": {
                "manifest_csv": "manifest.csv",
                "split_dir": "splits",
                "prediction_h5_dir": "predictions",
            },
            "features": {
                "root": "features",
                "he_aligned_file": "he.npy",
                "virtual_channels_file": "virtual_channels.npy",
                "virtual_aggregated_file": "virtual.npy",
                "patch_ids_file": "patch_ids.json",
                "extractor": {"name": "summary_stats", "batch_size": 2},
            },
            "split": {
                "folds": 2,
                "seed": 1,
                "stratify_column": "label" if task == "classification" else None,
                "unmatched_policy": "error",
            },
            "model": {
                "name": "amil",
                "input_modality": "virtual",
                "virtual_representation": "aggregated",
                "virtual_input_dim": 4,
                "n_time_bins": 4,
                "attention_dim": 4,
                "projection_dim": 4,
                "dropout": 0.0,
            },
            "train": {
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "alpha_survival": 0.0,
                "seed": 1,
            },
            "inference": {"split": "val"},
            "evaluation": {
                "bootstrap_iterations": 10,
                "bootstrap_seed": 3,
                "confidence_level": 0.95,
                "plot_km": False,
                "cox": False,
            },
            "runtime": {"device": "cpu"},
            "pipeline": {"stages": ["make-splits", "train", "infer", "evaluate"]},
            "output": {"run_dir": "run"},
        }
        path = root / f"{task}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _bags(self, root: Path, *, task: str = "survival") -> None:
        rows = []
        rng = np.random.default_rng(4)
        for index in range(8):
            patient_id = f"patient_{index}"
            slide_id = f"slide_{index}"
            row = {"patient_id": patient_id, "slide_id": slide_id}
            if task == "survival":
                row.update({"event_time": float(index + 1), "censor": int(index in {1, 5})})
            else:
                row["label"] = "positive" if index % 2 else "negative"
            rows.append(row)
            slide_dir = root / "features" / slide_id
            slide_dir.mkdir(parents=True)
            np.save(slide_dir / "virtual.npy", rng.normal(size=(4, 4)).astype(np.float32))
        pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)

    def test_harrell_c_index_direction(self) -> None:
        self.assertEqual(harrell_c_index([1, 1, 1], [1, 2, 3], [3, 2, 1]), 1.0)
        self.assertEqual(harrell_c_index([1, 1, 1], [1, 2, 3], [1, 2, 3]), 0.0)

    def test_survival_train_infer_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            self._bags(root)
            result = run_clinical_pipeline(config)
            self.assertEqual(result["stages"], ["make-splits", "train", "infer", "evaluate"])
            self.assertTrue((root / "run" / "checkpoints" / "fold_0_best.pt").is_file())
            self.assertTrue((root / "run" / "clinical_predictions.csv").is_file())
            summary = json.loads((root / "run" / "evaluation" / "summary.json").read_text())
            self.assertEqual(summary["primary_level"], "patient")
            self.assertEqual(summary["patient"]["n_folds"], 2)
            for fold in range(2):
                split = pd.read_csv(root / "splits" / f"splits_{fold}.csv")
                self.assertFalse(set(split["train"].dropna()) & set(split["val"].dropna()))

    def test_classification_train_infer_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root, task="classification")
            self._bags(root, task="classification")
            run_clinical_pipeline(config)
            summary = json.loads((root / "run" / "evaluation" / "summary.json").read_text())
            self.assertEqual(summary["task"], "classification")
            self.assertIn("auc", summary["patient"])

    def test_extract_summary_features_from_submission_h5(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            with h5py.File(prediction_dir / "slide_demo.h5", "w") as handle:
                metadata = handle.create_group("meta")
                metadata.attrs["slide_name"] = "slide_demo"
                data = handle.create_group("data")
                data.create_dataset("pred", data=np.arange(2 * 4 * 4 * 2, dtype=np.uint8).reshape(2, 4, 4, 2))
                data.create_dataset("rows", data=np.asarray([0, 0]))
                data.create_dataset("cols", data=np.asarray([0, 1]))
                data.create_dataset("channel_names", data=np.asarray([b"CD3", b"CD8"]))
            result = run_clinical_pipeline(config, stages=["extract-features"])
            stage = result["results"][0]
            self.assertEqual(stage["extractor"], "summary_stats")
            features = np.load(root / "features" / "slide_demo" / "virtual_channels.npy")
            self.assertEqual(features.shape, (2, 2, 4))


if __name__ == "__main__":
    unittest.main()
