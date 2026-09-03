import argparse
import json
import os
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.config import Config
from src.dataset.dataset_imc01 import TuProDatasetIMC01
from src.trainers.histoplexer_trainer import HistoplexerTrainer
from src.utils.misc import seed_everything


def _ensure_imc01_method(config: Config) -> None:
    if not str(config.method).endswith("_imc01"):
        config.method = f"{config.method}_imc01"


def main(args, device):
    seed_everything(seed=args.seed, device=device)

    train_dataset = TuProDatasetIMC01(
        split=args.train_csv or args.split,
        mode="train",
        src_folder=args.src_folder,
        tgt_folder=args.tgt_folder,
        use_high_res=args.use_high_res,
        p_flip_jitter_hed_affine=args.p_flip_jitter_hed_affine,
        patch_size=args.patch_size,
        channels=args.channels,
        cohort=args.cohort,
        use_fm_features=args.use_fm_features,
        fm_features_path=args.fm_features_path,
    )

    datasets = [train_dataset]

    if args.val:
        val_dataset = TuProDatasetIMC01(
            split=args.val_csv or args.split,
            mode="valid",
            src_folder=args.src_folder,
            tgt_folder=args.tgt_folder,
            use_high_res=args.use_high_res,
            p_flip_jitter_hed_affine=args.p_flip_jitter_hed_affine,
            patch_size=args.patch_size,
            channels=args.channels,
            cohort=args.cohort,
            use_fm_features=args.use_fm_features,
            fm_features_path=args.fm_features_path,
        )
        datasets.append(val_dataset)

    print(f"Number of training images: {len(train_dataset)}")

    trainer = HistoplexerTrainer(args=args, datasets=datasets)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurations for HistoPlexer (IMC scaled to 0-1)")
    parser.add_argument("--config_path", type=str, help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config_path, "r") as ifile:
        config = Config(json.load(ifile))

    _ensure_imc01_method(config)

    if config.resume_path is None:
        channel_str = "all" if config.channels is None else str(config.markers[config.channels[0]])
        config.experiment_name = f"{config.cohort}_{config.method}_channels-{channel_str}_seed-{config.seed}"
    else:
        config.experiment_name = config.resume_path.split("/")[-1]

    print(config.experiment_name)

    config.save_path = os.path.join(config.base_save_path, config.experiment_name)
    print(f"save path: {config.save_path}")
    Path(config.save_path).mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in {config.save_path}")
    print(f"Using device: {config.device}")

    with open(os.path.join(config.save_path, "config.json"), "w") as ofile:
        json.dump(config.__dict__, ofile, indent=4)

    main(config, device=torch.device(config.device))
    print("Done!")
