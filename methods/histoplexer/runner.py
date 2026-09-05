from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from methods.runner_utils import (
    deep_merge,
    emit_or_run,
    gpu_index,
    inference_payload,
    load_config,
    native_task,
    object_at,
    parser,
    python_command,
    resolve_known_paths,
    resolve_path,
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
    args = parser("HEPROBench adapter for the HistoPlexer pipeline").parse_args()
    config, source = load_config(args.config)
    destination = run_dir(config, args.task)

    if args.task == "train":
        task_cfg = native_task(config, "train")
        payload = task_cfg.get("config", {})
        if not isinstance(payload, dict):
            raise ValueError("native.train.config must be an object")
        payload = deep_merge(payload, object_at(config, "histoplexer"))
        data, train, output, runtime = (
            object_at(config, "data"),
            object_at(config, "train"),
            object_at(config, "output"),
            object_at(config, "runtime"),
        )
        for key in ("src_folder", "tgt_folder", "split", "vgg_path", "fm_features_path"):
            set_from(payload, key, data.get(key))
        set_from(payload, "train_csv", split_csv(data, "train"))
        set_from(payload, "val_csv", split_csv(data, "valid"))
        set_from(payload, "test_csv", split_csv(data, "test"))
        for key in ("batch_size", "num_workers", "patch_size", "seed", "total_steps", "log_interval", "save_interval"):
            set_from(payload, key, train.get(key), data.get(key))
        set_from(payload, "base_save_path", output.get("checkpoint_dir"), str(destination))
        set_from(payload, "device", runtime.get("device"))
        if payload.get("split"):
            payload["split"] = resolve_path(payload["split"], source)
        payload = resolve_known_paths(payload, source)
        native_path = write_json(payload, destination / "native_train_config.json")
        launcher = task_cfg.get("launcher", {})
        distributed = isinstance(launcher, dict) and str(launcher.get("type", "python")).lower() == "torchrun"
        if distributed:
            command = training_command(
                IMPLEMENTATION / "bin" / "train_ddp_imc01.py",
                ["--config_path", str(native_path)],
                config,
                task_cfg,
                module="bin.train_ddp_imc01",
            )
        else:
            command = python_command(
                IMPLEMENTATION / "bin" / "train_imc01.py",
                ["--config_path", str(native_path)],
                config,
            )
        return emit_or_run(command, cwd=IMPLEMENTATION, dry_run=args.dry_run, native_config=native_path)

    payload, _ = inference_payload(config, source, method_name="HistoPlexer")
    set_default(payload, "out_root", str(destination / "predictions"))
    set_default(payload, "gpu_id", gpu_index(object_at(config, "runtime").get("device")))
    native_path = write_json(payload, destination / "native_infer_config.json")
    command = python_command(IMPLEMENTATION / "sp_infer.py", ["--config", str(native_path)], config)
    return emit_or_run(command, cwd=IMPLEMENTATION, dry_run=args.dry_run, native_config=native_path)


if __name__ == "__main__":
    raise SystemExit(main())
