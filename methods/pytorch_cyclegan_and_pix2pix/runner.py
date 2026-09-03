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


def _variant_defaults(values: dict, variant: str) -> None:
    if variant == "pix2pix":
        set_default(values, "model", "pix2pix")
        set_default(values, "dataset_mode", "HE2SPREPEAT")
        set_default(values, "netG", "unet_256")
        set_default(values, "norm", "batch")
    elif variant == "cyclegan":
        set_default(values, "model", "cycle_gan")
        set_default(values, "dataset_mode", "HE2SPREPEAT")
        set_default(values, "netG", "resnet_9blocks")
        set_default(values, "norm", "instance")
    else:
        raise ValueError(f"Unsupported variant: {variant}")


def main() -> int:
    args = parser("HEPROBench adapter for Pix2Pix and CycleGAN", variant=True).parse_args()
    config, source = load_config(args.config)
    destination = run_dir(config, args.task)

    if args.task == "train":
        task_cfg = native_task(config, "train")
        values = task_cfg.get("args", {})
        if not isinstance(values, dict):
            raise ValueError("native.train.args must be an object")
        values = dict(values)
        _variant_defaults(values, args.variant)
        data, train, output = object_at(config, "data"), object_at(config, "train"), object_at(config, "output")
        set_default(values, "dataroot", data.get("root_dir"))
        set_default(values, "checkpoints_dir", output.get("checkpoint_dir"), str(destination / "checkpoints"))
        set_default(values, "name", output.get("run_name"), args.variant)
        set_default(values, "batch_size", train.get("batch_size"))
        set_default(values, "gpu_ids", gpu_index(object_at(config, "runtime").get("device")))
        values = resolve_known_paths(values, source)
        command = training_command(
            IMPLEMENTATION / "train.py",
            cli_args(values, task_cfg.get("extra_args", [])),
            config,
            task_cfg,
        )
        return emit_or_run(command, cwd=IMPLEMENTATION, dry_run=args.dry_run)

    payload, _ = inference_payload(config, source, method_name=args.variant)
    _variant_defaults(payload, args.variant)
    set_default(payload, "out_root", str(destination / "predictions"))
    set_default(payload, "gpu_ids", gpu_index(object_at(config, "runtime").get("device")))
    native_path = write_json(payload, destination / "native_infer_config.json")
    command = python_command(IMPLEMENTATION / "sp_infer.py", ["--config", str(native_path)], config)
    return emit_or_run(command, cwd=IMPLEMENTATION, dry_run=args.dry_run, native_config=native_path)


if __name__ == "__main__":
    raise SystemExit(main())
