from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .experiment import (
    get_run_dir,
    load_experiment_config,
    resolve_experiment_config,
    write_resolved_config,
)
from .registry import resolve_encoder, resolve_method


def dispatch_method_task(
    *,
    task: str,
    method_name: str,
    config_path: str | Path,
    encoder: str | None = None,
    split: str | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    output_dir: str | None = None,
    overrides: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    method = resolve_method(method_name)
    if task not in method.tasks:
        supported = ", ".join(sorted(method.tasks)) or "none"
        raise ValueError(f"Method '{method.name}' does not support task '{task}'. Supported: {supported}")
    if method.name == "dpt_fm":
        if not encoder:
            raw, _ = load_experiment_config(config_path)
            model = raw.get("model", {})
            encoder_block = model.get("encoder", {}) if isinstance(model, dict) else {}
            encoder = str(encoder_block.get("name", "")) if isinstance(encoder_block, dict) else ""
        if not encoder:
            raise ValueError("dpt_fm requires model.encoder.name in JSON or --encoder")
        resolve_encoder(encoder)

    raw_config, source_path = load_experiment_config(config_path)
    resolved = resolve_experiment_config(
        raw_config,
        source_path=source_path,
        method=method.name,
        task=task,
        encoder=encoder,
        split=split,
        device=device,
        batch_size=batch_size,
        output_dir=output_dir,
        overrides=overrides,
    )
    if method.name == "dpt_fm":
        resolved_model = resolved.get("model", {})
        resolved_encoder = resolved_model.get("encoder", {}) if isinstance(resolved_model, dict) else {}
        final_encoder = str(resolved_encoder.get("name", "")) if isinstance(resolved_encoder, dict) else ""
        if not final_encoder:
            raise ValueError("dpt_fm requires model.encoder.name in JSON or --encoder")
        resolve_encoder(final_encoder)
        encoder = final_encoder
    run_dir = get_run_dir(resolved, method=method.name, task=task, source_path=source_path)
    resolved_path = write_resolved_config(resolved, run_dir, task)

    task_spec = method.tasks[task]
    if not task_spec.entrypoint.exists():
        raise FileNotFoundError(
            f"Registered entrypoint is missing for {method.name}.{task}: {task_spec.entrypoint}"
        )
    runtime = resolved.get("runtime", {})
    if runtime is not None and not isinstance(runtime, dict):
        raise ValueError("'runtime' must be a JSON object")
    python_executable = str((runtime or {}).get("python") or sys.executable)
    command = [
        python_executable,
        str(task_spec.entrypoint),
        *task_spec.args,
        task,
        "--config",
        str(resolved_path),
    ]
    result: dict[str, Any] = {
        "method": method.name,
        "task": task,
        "encoder": encoder or "",
        "config": str(resolved_path),
        "run_dir": str(run_dir),
        "command": shlex.join(command),
        "dry_run": bool(dry_run),
    }
    env = os.environ.copy()
    env.update(
        {
            "HEPROBENCH_METHOD": method.name,
            "HEPROBENCH_TASK": task,
            "HEPROBENCH_RUN_DIR": str(run_dir),
        }
    )
    if dry_run:
        command.append("--dry-run")
        result["command"] = shlex.join(command)
        completed = subprocess.run(
            command,
            cwd=str(task_spec.entrypoint.parent),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        try:
            result["native"] = __import__("json").loads(completed.stdout)
        except ValueError:
            result["native"] = {"stdout": completed.stdout.strip()}
        return result

    completed = subprocess.run(command, cwd=str(task_spec.entrypoint.parent), env=env, check=False)
    result["returncode"] = int(completed.returncode)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return result
