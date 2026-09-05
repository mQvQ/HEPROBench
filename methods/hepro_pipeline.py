"""Adapter shared by MIPHEI-ViT, GigaTIME-Reg, and DPT+FM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from methods.runner_utils import (
    deep_merge,
    emit_or_run,
    gpu_index,
    inference_payload,
    native_task,
    object_at,
    python_command,
    resolve_known_paths,
    set_default,
    write_json,
    write_yaml_compatible_json,
)


IMPLEMENTATION = Path(__file__).resolve().parent / "miphei_vit" / "implementation" / "hepro"


def _model_config(config: dict[str, Any], variant: str, encoder: str | None) -> dict[str, Any]:
    raw = object_at(config, "model")
    result = {key: value for key, value in raw.items() if key not in {"checkpoint", "checkpoint_dir"}}
    if variant == "dpt_fm":
        result["model_name"] = "dpt"
        result["use_lora"] = False
        encoder_cfg = result.get("encoder", {})
        if not isinstance(encoder_cfg, dict):
            raise ValueError("model.encoder must be an object")
        encoder_cfg = dict(encoder_cfg)
        encoder_cfg["encoder_name"] = encoder or encoder_cfg.get("name") or encoder_cfg.get("encoder_name")
        encoder_cfg.pop("name", None)
        encoder_cfg.setdefault("encoder_weights", None)
        encoder_cfg["frozen"] = True
        result["encoder"] = encoder_cfg
    elif variant == "miphei_vit":
        result.setdefault("model_name", "myvitmatte")
        result.setdefault("use_lora", True)
        encoder_cfg = result.get("encoder", {})
        if isinstance(encoder_cfg, dict):
            encoder_cfg = dict(encoder_cfg)
            if "name" in encoder_cfg:
                encoder_cfg["encoder_name"] = encoder_cfg.pop("name")
            result["encoder"] = encoder_cfg
    elif variant == "gigatime_reg":
        result["model_name"] = "gigatime"
        result.setdefault("use_lora", False)
    else:
        raise ValueError(f"Unknown HEPRO variant: {variant}")
    return result


def run_hepro(
    *,
    config: dict[str, Any],
    source: Path,
    destination: Path,
    task: str,
    variant: str,
    encoder: str | None,
    dry_run: bool,
) -> int:
    task_cfg = native_task(config, task)
    if task == "train":
        payload = task_cfg.get("config", {})
        if not isinstance(payload, dict):
            raise ValueError("native.train.config must be an object")
        payload = deep_merge(payload, {"data": object_at(config, "data")})
        payload = deep_merge(payload, {"train": object_at(config, "train")})
        model = object_at(config, "model")
        payload = deep_merge(payload, {"model": _model_config(config, variant, encoder)})
        payload = deep_merge(payload, {"runtime": object_at(config, "runtime")})
        output = {**object_at(config, "output"), "run_dir": str(destination)}
        output.setdefault("checkpoint_dir", model.get("checkpoint_dir") or str(destination / "checkpoint"))
        payload = deep_merge(payload, {"output": output})
        payload = resolve_known_paths(payload, source)
        native_path = write_yaml_compatible_json(payload, destination / "native_train_config.yaml")
        extra = task_cfg.get("hydra_overrides", [])
        if not isinstance(extra, list):
            raise ValueError("native.train.hydra_overrides must be a list")
        command = python_command(
            IMPLEMENTATION / "run.py",
            [
                "--config-path", str(native_path.parent),
                "--config-name", native_path.stem,
                f"hydra.run.dir={destination / 'hydra'}",
                "hydra.job.chdir=false",
                *[str(item) for item in extra],
            ],
            config,
        )
        return emit_or_run(command, cwd=IMPLEMENTATION, dry_run=dry_run, native_config=native_path)

    names = {"miphei_vit": "MIPHEI-ViT", "gigatime_reg": "GigaTIME-Reg", "dpt_fm": f"DPT-{encoder}"}
    payload, _ = inference_payload(config, source, method_name=names[variant])
    set_default(payload, "checkpoint_dir", str(destination / "checkpoint"))
    dataset_config = payload.pop("dataset_config", None)
    if dataset_config is None and not payload.get("dataset_config_path"):
        dataset_config = {"data": object_at(config, "data")}
    if dataset_config is not None:
        if not isinstance(dataset_config, dict):
            raise ValueError("native.infer.config.dataset_config must be an object")
        dataset_path = write_yaml_compatible_json(
            resolve_known_paths(dataset_config, source),
            destination / "native_dataset_config.yaml",
        )
        payload["dataset_config_path"] = str(dataset_path)
    set_default(payload, "out_root", str(destination / "predictions"))
    set_default(payload, "gpu_id", gpu_index(object_at(config, "runtime").get("device")))
    payload = resolve_known_paths(payload, source)
    native_path = write_json(payload, destination / "native_infer_config.json")
    command = python_command(IMPLEMENTATION / "sp_infer.py", ["--config", str(native_path)], config)
    return emit_or_run(command, cwd=IMPLEMENTATION, dry_run=dry_run, native_config=native_path)
