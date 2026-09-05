from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from heprobench.dispatch import dispatch_method_task
from heprobench.experiment import load_experiment_config
from heprobench.registry import load_encoder_registry, load_method_registry
from heprobench.profile import profile


ROOT = Path(__file__).resolve().parents[1]


class RegistryTests(unittest.TestCase):
    def test_bundled_demo_has_train_valid_test_data(self) -> None:
        expected_counts = {"train": 4, "valid": 2, "test": 2}
        root = ROOT / "demo_data"
        channel_names = json.loads((root / "channel_names.json").read_text(encoding="utf-8"))
        self.assertEqual(channel_names, ["DAPI", "CD3", "CD20", "PanCK"])
        self.assertTrue((root / "channel_stats.json").is_file())
        for split, expected_count in expected_counts.items():
            with self.subTest(split=split):
                with (root / "splits" / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), expected_count)
                for row in rows:
                    self.assertTrue((root / row["image_path"]).is_file())
                    self.assertTrue((root / row["target_path"]).is_file())
                    self.assertTrue((root / row["mask_path"]).is_file())
                    mask = np.load(root / row["mask_path"])
                    self.assertEqual(mask.shape, (256, 256))
                    self.assertTrue(np.issubdtype(mask.dtype, np.integer))
                    self.assertGreater(int(mask.max()), 0)
                    self.assertGreaterEqual(int(mask.min()), 0)
        with (root / "cell_annotations.csv").open(newline="", encoding="utf-8") as handle:
            cell_rows = list(csv.DictReader(handle))
        self.assertEqual(len(cell_rows), 128)
        self.assertEqual({row["split"] for row in cell_rows}, {"train", "valid", "test"})
        self.assertEqual(
            {split: sum(row["split"] == split for row in cell_rows) for split in expected_counts},
            {"train": 64, "valid": 32, "test": 32},
        )
        cell_keys = {(row["slide_name"], int(row["global_cell_id"])) for row in cell_rows}
        self.assertEqual(len(cell_keys), len(cell_rows))

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
        "histoplexer": ("configs/experiments/histoplexer.json", "bin.train_ddp_imc01"),
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

    def test_json_method_name_is_sufficient_and_shared_batch_size_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = dispatch_method_task(
                task="train",
                method_name=None,
                config_path=ROOT / "configs" / "experiments" / "rosie.json",
                output_dir=temp_dir,
                overrides=["train.batch_size=3"],
                dry_run=True,
            )
        self.assertEqual(result["method"], "rosie")
        self.assertIn("--batch_size 3", result["native"]["command"])
        self.assertNotIn("--batch_size 64", result["native"]["command"])

    def test_demo_configs_use_bundled_split_csvs(self) -> None:
        demo_configs = sorted((ROOT / "configs" / "demo").glob("*.json"))
        demo_configs = [path for path in demo_configs if not path.name.startswith("_")]
        self.assertEqual(len(demo_configs), 10)
        with tempfile.TemporaryDirectory() as temp_dir:
            for config_path in demo_configs:
                with self.subTest(config=config_path.name):
                    result = dispatch_method_task(
                        task="train",
                        method_name=None,
                        config_path=config_path,
                        output_dir=str(Path(temp_dir) / config_path.stem),
                        dry_run=True,
                    )
                    self.assertTrue(result["native"]["command"])

    def test_demo_inference_configs_match_four_channel_bundle(self) -> None:
        demo_configs = sorted((ROOT / "configs" / "demo").glob("*.json"))
        demo_configs = [path for path in demo_configs if not path.name.startswith("_")]
        with tempfile.TemporaryDirectory() as temp_dir:
            for config_path in demo_configs:
                with self.subTest(config=config_path.name):
                    result = dispatch_method_task(
                        task="infer",
                        method_name=None,
                        config_path=config_path,
                        output_dir=str(Path(temp_dir) / config_path.stem),
                        dry_run=True,
                    )
                    native_path = result["native"].get("native_config")
                    self.assertTrue(native_path)
                    payload = json.loads(Path(native_path).read_text(encoding="utf-8"))
                    self.assertEqual(payload["split"], "test")
                    if "output_nc" in payload:
                        self.assertEqual(payload["output_nc"], 4)

    def test_demo_training_checkpoints_are_consumed_by_inference(self) -> None:
        hepro_stems = ("gigatime_reg", "miphei_vit", "dpt_fm_h0-mini")
        with tempfile.TemporaryDirectory() as temp_dir:
            for stem in hepro_stems:
                with self.subTest(method=stem):
                    config_path = ROOT / "configs" / "demo" / f"{stem}.json"
                    train_result = dispatch_method_task(
                        task="train",
                        method_name=None,
                        config_path=config_path,
                        output_dir=str(Path(temp_dir) / stem),
                        dry_run=True,
                    )
                    infer_result = dispatch_method_task(
                        task="infer",
                        method_name=None,
                        config_path=config_path,
                        output_dir=str(Path(temp_dir) / stem),
                        dry_run=True,
                    )
                    train_payload = json.loads(
                        Path(train_result["native"]["native_config"]).read_text(encoding="utf-8")
                    )
                    infer_payload = json.loads(
                        Path(infer_result["native"]["native_config"]).read_text(encoding="utf-8")
                    )
                    self.assertEqual(train_payload["output"]["checkpoint_dir"], infer_payload["checkpoint_dir"])

        hex_config, _ = load_experiment_config(ROOT / "configs" / "demo" / "hex.json")
        self.assertEqual(hex_config["train"]["ckpt_interval"], 1)
        self.assertTrue(hex_config["model"]["checkpoint_path"].endswith("checkpoint_step_1.pth"))

    def test_shared_batch_size_reaches_each_native_pipeline(self) -> None:
        direct_args = {
            "rosie": "--batch_size 3",
            "hex": "--batch_size_per_gpu 3",
            "cut": "--batch_size 3",
            "pix2pix": "--batch_size 3",
            "cyclegan": "--batch_size 3",
            "gigatime_original": "--batch_size 3",
        }
        config_backed = {
            "histoplexer": ("batch_size",),
            "gigatime_reg": ("train", "batch_size"),
            "miphei_vit": ("train", "batch_size"),
            "dpt_fm_h0-mini": ("train", "batch_size"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for stem, expected_arg in direct_args.items():
                with self.subTest(method=stem):
                    result = dispatch_method_task(
                        task="train",
                        method_name=None,
                        config_path=ROOT / "configs" / "demo" / f"{stem}.json",
                        output_dir=str(Path(temp_dir) / stem),
                        batch_size=3,
                        dry_run=True,
                    )
                    self.assertIn(expected_arg, result["native"]["command"])

            for stem, keys in config_backed.items():
                with self.subTest(method=stem):
                    result = dispatch_method_task(
                        task="train",
                        method_name=None,
                        config_path=ROOT / "configs" / "demo" / f"{stem}.json",
                        output_dir=str(Path(temp_dir) / stem),
                        batch_size=3,
                        dry_run=True,
                    )
                    payload = json.loads(Path(result["native"]["native_config"]).read_text(encoding="utf-8"))
                    value = payload
                    for key in keys:
                        value = value[key]
                    self.assertEqual(value, 3)

    def test_formal_defaults_match_recorded_training_runs(self) -> None:
        rosie, _ = load_experiment_config(ROOT / "configs" / "experiments" / "rosie.json")
        self.assertEqual(rosie["train"]["batch_size"], 8)
        self.assertEqual(rosie["data"]["patch_size"], 128)
        self.assertEqual(rosie["train"]["samples_per_image"], 16)

        cut, _ = load_experiment_config(ROOT / "configs" / "experiments" / "cut.json")
        self.assertEqual(cut["train"]["batch_size"], 8)

        cycle, _ = load_experiment_config(ROOT / "configs" / "experiments" / "cyclegan.json")
        self.assertEqual(cycle["native"]["train"]["args"]["input_nc"], 3)
        self.assertEqual(cycle["native"]["train"]["args"]["dataset_mode"], "HE2SP")

        histo, _ = load_experiment_config(ROOT / "configs" / "experiments" / "histoplexer.json")
        self.assertEqual(histo["train"]["batch_size"], 8)
        self.assertEqual(histo["train"]["seed"], 96)
        self.assertEqual(histo["native"]["train"]["launcher"]["nproc_per_node"], 2)
        self.assertEqual(histo["native"]["train"]["config"]["output_nc"], 60)
        self.assertEqual(histo["native"]["infer"]["config"]["output_nc"], 60)

    def test_every_demo_config_has_a_native_efficiency_profile(self) -> None:
        demo_configs = sorted((ROOT / "configs" / "demo").glob("*.json"))
        demo_configs = [path for path in demo_configs if not path.name.startswith("_")]
        with tempfile.TemporaryDirectory() as temp_dir:
            for config_path in demo_configs:
                with self.subTest(config=config_path.name):
                    result = profile(
                        config_path,
                        Path(temp_dir) / config_path.stem,
                        device="cpu",
                        dry_run=True,
                    )
                    self.assertEqual(result["schema"], "heprobench_efficiency_v1")
                    self.assertEqual(len(result["commands"]), 2)
                    self.assertTrue(all(Path(command.split()[1]).is_file() for command in result["commands"]))


if __name__ == "__main__":
    unittest.main()
