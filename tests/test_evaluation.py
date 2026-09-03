from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from heprobench.config import load_dataset_config
from heprobench.evaluate import evaluate
from heprobench.io_h5 import write_slide_h5
from heprobench.schema import group_by_slide, load_channel_names, load_records


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "demo" / "rosie.json"


class EvaluationTests(unittest.TestCase):
    def test_demo_runs_slide_and_cell_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_root = Path(temp_dir) / "predictions"
            efficiency_path = Path(temp_dir) / "efficiency.json"
            efficiency_path.write_text(
                json.dumps(
                    {
                        "schema": "heprobench_efficiency_v1",
                        "enabled": True,
                        "method": "rosie",
                        "params_total": 1,
                        "gflops_per_patch": 0.1,
                        "latency_mean_ms_per_region": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            for split in ("valid", "test"):
                cfg, _ = load_dataset_config(CONFIG, split_override=split)
                dataset = cfg["dataset"]
                channels = load_channel_names(dataset["channel_names"])
                for slide_name, records in group_by_slide(load_records(dataset)).items():
                    predictions = [np.load(record.target_path) for record in records]
                    write_slide_h5(
                        pred_root / split / f"{slide_name}.h5",
                        np.stack(predictions),
                        records,
                        channels,
                        {"method_name": "test", "split": split, "patch_size": 256},
                    )

            summary = evaluate(
                CONFIG,
                pred_root,
                perceptual=False,
                efficiency_json=efficiency_path,
            )
            result_dir = pred_root / "test"
            self.assertEqual(summary["primary_aggregation"], "slide_macro")
            self.assertEqual(summary["n_slides"], 1)
            self.assertIn("rmse", summary["image_metrics"])
            self.assertNotIn("lpips", summary["image_metrics"])
            self.assertTrue(summary["computational_efficiency"]["enabled"])
            self.assertTrue(summary["cell"]["enabled"])
            self.assertEqual(summary["cell"]["n_valid_cells"], 32)
            self.assertEqual(summary["cell"]["n_test_cells"], 32)
            for name in (
                "metrics_tile_channel.csv",
                "metrics_slide_channel.csv",
                "metrics_slide.csv",
                "cell_metrics/cell_pcc_per_slide_marker.csv",
                "cell_metrics/cell_classification_per_marker.csv",
                "cell_metrics/cell_classification_per_slide_marker.csv",
            ):
                self.assertTrue((result_dir / name).is_file(), name)
            payload = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["cell"]["classification"]["n_markers"], 3)
            self.assertIsNone(payload["psnr"])


if __name__ == "__main__":
    unittest.main()
