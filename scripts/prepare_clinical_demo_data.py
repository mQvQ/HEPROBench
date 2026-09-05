from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deidentified synthetic bags for the clinical demo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "clinical_demo" / "data",
    )
    parser.add_argument("--patients", type=int, default=24)
    parser.add_argument("--patches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.patients < 8:
        raise ValueError("Clinical demo needs at least eight synthetic patients")

    output_dir = args.output_dir.expanduser().resolve()
    feature_root = output_dir / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for patient_index in range(args.patients):
        patient_id = f"synthetic_patient_{patient_index:03d}"
        slide_id = f"synthetic_slide_{patient_index:03d}"
        latent_risk = float(rng.normal())
        event_time = float(np.clip(54.0 - 11.0 * latent_risk + rng.normal(scale=3.0), 4.0, 96.0))
        censor = int(patient_index % 4 == 0)
        rows.append(
            {
                "patient_id": patient_id,
                "slide_id": slide_id,
                "event_time": event_time,
                "censor": censor,
                "label": "positive" if patient_index % 2 else "negative",
            }
        )

        he = rng.normal(size=(args.patches, 8)).astype(np.float32)
        he[:, 0] += 0.35 * latent_risk
        virtual = rng.normal(scale=0.6, size=(args.patches, 4, 6)).astype(np.float32)
        virtual[:, :, 0] += latent_risk
        virtual[:, 0, 1] += 0.5 * latent_risk
        slide_dir = feature_root / slide_id
        slide_dir.mkdir(parents=True, exist_ok=True)
        np.save(slide_dir / "he_aligned.npy", he)
        np.save(slide_dir / "virtual_channels.npy", virtual)
        np.save(slide_dir / "virtual_aggregated.npy", virtual.mean(axis=1))
        (slide_dir / "patch_ids.json").write_text(
            json.dumps([f"patch_{index:03d}" for index in range(args.patches)], indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(
        json.dumps(
            {
                "schema": "heprobench_synthetic_clinical_demo_v1",
                "manifest": str(manifest),
                "feature_root": str(feature_root),
                "patients": args.patients,
                "patches_per_slide": args.patches,
                "contains_real_data": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
