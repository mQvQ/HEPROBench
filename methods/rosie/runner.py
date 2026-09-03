from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
IMPLEMENTATION = HERE / "implementation"


def main() -> int:
    args = parser("HEPROBench adapter for the ROSIE pipeline").parse_args()
    config, source = load_config(args.config)
    destination = run_dir(config, args.task)

    if args.task == "train":
        task_cfg = native_task(config, "train")
        values = task_cfg.get("args", {})
        if not isinstance(values, dict):
            raise ValueError("native.train.args must be an object")
        values = dict(values)
        data, train, output = object_at(config, "data"), object_at(config, "train"), object_at(config, "output")
        set_default(values, "root_dir", data.get("root_dir"))
        set_default(values, "checkpoints_dir", output.get("checkpoint_dir"), str(destination / "checkpoints"))
        for key in ("batch_size", "learning_rate", "patch_size", "output_nc", "total_iteration", "save_per_iteration", "eval_interval", "num_workers", "samples_per_image", "seed", "preprocess"):
            set_default(values, key, train.get(key), data.get(key))
        set_default(values, "name", output.get("run_name"))
        values = resolve_known_paths(values, source)
        command = training_command(
            IMPLEMENTATION / "train_he2sp.py",
            cli_args(values, task_cfg.get("extra_args", [])),
            config,
            task_cfg,
        )
        return emit_or_run(command, cwd=IMPLEMENTATION, dry_run=args.dry_run)

    payload, _ = inference_payload(config, source, method_name="ROSIE")
    set_default(payload, "out_root", str(destination / "predictions"))
    device = object_at(config, "runtime").get("device")
    set_default(payload, "gpu_id", gpu_index(device))
    native_path = write_json(payload, destination / "native_infer_config.json")
    command = python_command(IMPLEMENTATION / "sp_infer.py", ["--config", str(native_path)], config)
    return emit_or_run(command, cwd=IMPLEMENTATION, dry_run=args.dry_run, native_config=native_path)


if __name__ == "__main__":
    raise SystemExit(main())
