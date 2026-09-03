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
    set_from,
    split_csv,
    training_command,
    write_json,
)


HERE = Path(__file__).resolve().parent
IMPLEMENTATION = HERE / "implementation"


def main() -> int:
    args = parser("HEPROBench adapter for the CUT pipeline").parse_args()
    config, source = load_config(args.config)
    destination = run_dir(config, args.task)

    if args.task == "train":
        task_cfg = native_task(config, "train")
        values = task_cfg.get("args", {})
        if not isinstance(values, dict):
            raise ValueError("native.train.args must be an object")
        values = dict(values)
        data, train, output = object_at(config, "data"), object_at(config, "train"), object_at(config, "output")
        set_from(values, "dataroot", data.get("root_dir"))
        set_from(values, "train_csv", split_csv(data, "train"))
        set_from(values, "val_csv", split_csv(data, "valid"))
        set_from(values, "test_csv", split_csv(data, "test"))
        set_from(values, "checkpoints_dir", output.get("checkpoint_dir"), str(destination / "checkpoints"))
        set_from(values, "name", output.get("run_name"))
        mappings = {
            "batch_size": "batch_size",
            "num_threads": "num_workers",
            "lr": "learning_rate",
            "total_iterations": "total_iterations",
            "save_latest_freq": "save_latest_freq",
            "save_per_iteration": "save_per_iteration",
        }
        for native_key, shared_key in mappings.items():
            set_from(values, native_key, train.get(shared_key))
        set_from(values, "gpu_ids", gpu_index(object_at(config, "runtime").get("device")))
        values = resolve_known_paths(values, source)
        command = training_command(
            IMPLEMENTATION / "train.py",
            cli_args(values, task_cfg.get("extra_args", [])),
            config,
            task_cfg,
        )
        return emit_or_run(command, cwd=IMPLEMENTATION, dry_run=args.dry_run)

    payload, _ = inference_payload(config, source, method_name="CUT")
    set_default(payload, "out_root", str(destination / "predictions"))
    set_default(payload, "gpu_ids", gpu_index(object_at(config, "runtime").get("device")))
    native_path = write_json(payload, destination / "native_infer_config.json")
    command = python_command(IMPLEMENTATION / "sp_infer.py", ["--config", str(native_path)], config)
    return emit_or_run(command, cwd=IMPLEMENTATION, dry_run=args.dry_run, native_config=native_path)


if __name__ == "__main__":
    raise SystemExit(main())
