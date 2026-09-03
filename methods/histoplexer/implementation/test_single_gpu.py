#!/usr/bin/env python3
"""
Test script to verify single GPU training still works with modified code
"""

import torch
import json
from pathlib import Path
import tempfile
import os

def test_single_gpu_training():
    """Test that single GPU training initialization works"""
    print("Testing single GPU training compatibility...")

    # Import the modules we need
    from src.config.config import Config
    from src.trainers.histoplexer_trainer import HistoplexerTrainer

    # Create a minimal config
    config_dict = {
        "src_folder": "/tmp",
        "tgt_folder": "/tmp",
        "split": "/tmp/split.csv",
        "markers": ["CD16", "CD20"],
        "cohort": "test",
        "use_gp": True,
        "w_GP": 5.0,
        "w_ASP": 0.0,
        "batch_size": 2,
        "output_nc": 2,
        "device": "cpu",  # Use CPU for testing
        "channels": None,
        "seed": 42,
        "method": "test",
        "resume_path": None,
        "base_save_path": "/tmp",
        "ddp": {"enabled": False}
    }

    config = Config(config_dict)

    # Verify DDP is disabled
    assert config.ddp_enabled == False, "DDP should be disabled for single GPU test"
    print("✓ DDP correctly disabled")

    # Create mock dataset (we won't actually load data)
    class MockDataset:
        def __init__(self, size=10):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            return {
                "he_patch": torch.randn(3, 256, 256),
                "imc_patch": torch.randn(2, 256, 256),
                "fm_features": None,
                "sample": f"sample_{idx}"
            }

    # Create mock datasets
    train_dataset = MockDataset(10)
    val_dataset = MockDataset(5)
    datasets = [train_dataset, val_dataset]

    try:
        # Try to initialize trainer (this should work with our modifications)
        trainer = HistoplexerTrainer(args=config, datasets=datasets)

        # Check that trainer is properly initialized for single GPU
        assert trainer.rank == 0, f"Expected rank 0, got {trainer.rank}"
        assert trainer.world_size == 1, f"Expected world_size 1, got {trainer.world_size}"
        assert trainer.is_ddp == False, f"Expected is_ddp False, got {trainer.is_ddp}"

        print("✓ Trainer initialized correctly for single GPU")

        # Check that models are not wrapped in DDP
        assert not hasattr(trainer.G, 'module'), "Generator should not be wrapped in DDP"
        assert not hasattr(trainer.D, 'module'), "Discriminator should not be wrapped in DDP"

        print("✓ Models correctly not wrapped in DDP")

        # Check DataLoader type
        assert not hasattr(trainer.train_loader, 'sampler') or \
               not hasattr(trainer.train_loader.sampler, 'num_replicas'), \
               "DataLoader should not use DistributedSampler"

        print("✓ DataLoader correctly uses regular sampler")

        print("✓ Single GPU training compatibility test passed!")
        return True

    except Exception as e:
        print(f"✗ Single GPU training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test loading single GPU config"""
    print("Testing single GPU config loading...")

    try:
        config_path = "src/config/sample_config_single.json"
        with open(config_path, "r") as f:
            config = Config(json.load(f))

        assert config.ddp_enabled == False, "DDP should be disabled"
        assert config.device == "cuda:0", f"Expected device cuda:0, got {config.device}"

        print("✓ Single GPU config loaded correctly")
        return True

    except Exception as e:
        print(f"✗ Config loading test failed: {e}")
        return False


def main():
    print("Testing single GPU compatibility after DDP modifications...\n")

    success = True

    # Test config loading
    if not test_config_loading():
        success = False

    # Test trainer initialization
    if not test_single_gpu_training():
        success = False

    if success:
        print("\n🎉 All tests passed! Single GPU training is fully compatible.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")

    return success


if __name__ == "__main__":
    main()
