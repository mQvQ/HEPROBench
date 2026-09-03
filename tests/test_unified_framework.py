from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from heprobench.dispatch import dispatch_method_task
from heprobench.experiment import load_experiment_config
from heprobench.registry import load_encoder_registry, load_method_registry


ROOT = Path(__file__).resolve().parents[1]


class RegistryTests(unittest.TestCase):
    def test_method_registry_has_distinct_gigatime_variants(self) -> None:
        methods = load_method_registry()
        self.assertEqual(len(methods), 10)
        self.assertIn("gigatime_original", methods)
        self.assertIn("gigatime_reg", methods)
        for method in methods.values():
            self.assertEqual(set(method.tasks), {"train", "infer"})
            for task in method.tasks.values():
                self.assertTrue(task.entrypoint.is_file())

    def test_foundation_registry_is_exactly_15_without_musk(self) -> None:
        encoders = load_encoder_registry()
        self.assertEqual(len(encoders), 15)
        self.assertNotIn("musk", encoders)
        for encoder in encoders.values():
            self.assertEqual(encoder.benchmark_input_size, (256, 256))
            self.assertEqual(len(encoder.feature_layers), 4)
            self.assertIn("provider", encoder.checkpoint)

    def test_foundation_json_inheritance(self) -> None:
        config, source = load_experiment_config(ROOT / "configs" / "foundation_models" / "conch.json")
        self.assertEqual(source.name, "conch.json")
        self.assertEqual(config["method"]["name"], "dpt_fm")
        self.assertEqual(config["model"]["encoder"]["name"], "conch")
        self.assertTrue(config["model"]["encoder"]["frozen"])
        self.assertEqual(config["protocol"]["decoder"], "DPT")


class DryRunTests(unittest.TestCase):
    CASES = {
        "rosie": ("configs/experiments/rosie.json", "train_he2sp.py"),
        "hex": ("configs/experiments/hex.json", "run_train_dist_sp_fds_paper_norm01.py"),
        "cut": ("configs/experiments/cut.json", "train.py"),
        "pix2pix": ("configs/experiments/pix2pix.json", "train.py"),
        "cyclegan": ("configs/experiments/cyclegan.json", "train.py"),
        "histoplexer": ("configs/experiments/histoplexer.json", "bin.train_ddp"),
        "gigatime_original": ("configs/experiments/gigatime_original.json", "db_train.py"),
        "gigatime_reg": ("configs/experiments/gigatime_reg.json", "run.py"),
        "miphei_vit": ("configs/experiments/miphei_vit.json", "run.py"),
        "dpt_fm": ("configs/foundation_models/uni.json", "run.py"),
    }

    def test_every_training_runner_resolves_native_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for method, (relative_config, entrypoint) in self.CASES.items():
                with self.subTest(method=method):
                    result = dispatch_method_task(
                        task="train",
                        method_name=method,
                        config_path=ROOT / relative_config,
                        output_dir=str(Path(temp_dir) / method),
                        dry_run=True,
                    )
                    self.assertTrue(result["dry_run"])
                    self.assertIn(entrypoint, result["native"]["command"])

    def test_all_foundation_configs_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            configs = sorted((ROOT / "configs" / "foundation_models").glob("*.json"))
            configs = [path for path in configs if path.name != "dpt_fm_base.json"]
            self.assertEqual(len(configs), 15)
            for config_path in configs:
                with self.subTest(encoder=config_path.stem):
                    result = dispatch_method_task(
                        task="infer",
                        method_name="dpt_fm",
                        config_path=config_path,
                        output_dir=str(Path(temp_dir) / config_path.stem),
                        dry_run=True,
                    )
                    self.assertEqual(result["encoder"], config_path.stem)
                    native_path = Path(result["native"]["native_config"])
                    payload = json.loads(native_path.read_text(encoding="utf-8"))
                    self.assertEqual(payload["method_name"], f"DPT-{config_path.stem}")

    def test_unregistered_encoder_cannot_enter_through_set_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(KeyError):
                dispatch_method_task(
                    task="train",
                    method_name="dpt_fm",
                    config_path=ROOT / "configs" / "foundation_models" / "uni.json",
                    output_dir=temp_dir,
                    overrides=['model.encoder.name="musk"'],
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
