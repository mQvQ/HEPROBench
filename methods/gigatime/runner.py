from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from methods.hepro_pipeline import run_hepro
from methods.runner_utils import (
    cli_args,
    emit_or_run,
    gpu_index,
    inference_payload,
    load_config,
    native_task,
    object_at,
    parser,
    python_command,
    resolve_known_paths,
    run_dir,
    set_default,
    training_command,
    write_json,
)


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "implementation" / "original"


def _run_original(config: dict, source: Path, destination: Path, task: str, dry_run: bool) -> int:
    if task == "train":
        task_cfg = native_task(config, "train")
        values = task_cfg.get("args", {})
        if not isinstance(values, dict):
            raise ValueError("native.train.args must be an object")
        values = dict(values)
        data, train, output = object_at(config, "data"), object_at(config, "train"), object_at(config, "output")
        set_default(values, "metadata", data.get("metadata"), data.get("csv_path"))
        set_default(values, "tiling_dir", data.get("root_dir"))
        set_default(values, "output_dir", output.get("checkpoint_dir"), str(destination))
        for key in ("batch_size", "epochs", "num_workers", "lr", "weight_decay", "window_size"):
            set_default(values, key, train.get(key), data.get(key))
        device = object_at(config, "runtime").get("device")
        index = gpu_index(device)
        if index not in (None, "-1"):
            set_default(values, "gpu_ids", [int(item) for item in index.split(",")])
        values = resolve_known_paths(values, source)
        command = training_command(
            ORIGINAL / "scripts" / "db_train.py",
            cli_args(values, task_cfg.get("extra_args", [])),
            config,
            task_cfg,
        )
        return emit_or_run(command, cwd=ORIGINAL, dry_run=dry_run)

    payload, _ = inference_payload(config, source, method_name="GigaTIME-Original")
    model_config = payload.pop("model_config", None)
    if model_config is not None:
        if not isinstance(model_config, dict):
            raise ValueError("native.infer.config.model_config must be an object")
        model_config_path = write_json(
            model_config,
            destination / "native_model_config.json",
        )
        payload["model_config_path"] = str(model_config_path)
    set_default(payload, "out_root", str(destination / "predictions"))
    set_default(payload, "gpu_id", gpu_index(object_at(config, "runtime").get("device")))
    native_path = write_json(payload, destination / "native_infer_config.json")
    command = python_command(ORIGINAL / "sp_infer_pretrain.py", ["--config", str(native_path)], config)
    return emit_or_run(command, cwd=ORIGINAL, dry_run=dry_run, native_config=native_path)


def main() -> int:
    args = parser("HEPROBench adapters for original and regression GigaTIME", variant=True).parse_args()
    config, source = load_config(args.config)
    destination = run_dir(config, args.task)
    if args.variant == "original":
        return _run_original(config, source, destination, args.task, args.dry_run)
    if args.variant == "regression":
        return run_hepro(
            config=config,
            source=source,
            destination=destination,
            task=args.task,
            variant="gigatime_reg",
            encoder=None,
            dry_run=args.dry_run,
        )
    raise ValueError(f"Unsupported GigaTIME variant: {args.variant}")


if __name__ == "__main__":
    raise SystemExit(main())
