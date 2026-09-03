"""
General-purpose test/inference script for image-to-image translation.
This script is optimized for batch processing and parallel image saving.
"""
import os
import concurrent.futures
from pathlib import Path

import pyvips
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


# --- Helper function for saving images in parallel ---
def save_image(tensor, path):
    """
    Converts a single tensor to a numpy array and saves it as a TIFF image using pyvips.
    This function is designed to be called from multiple threads.
    """
    try:
        # Move tensor to CPU and convert to numpy array. The tensor is expected in CHW format.
        image_array = tensor

        # Create pyvips image and write to file with fast compression.
        image = pyvips.Image.new_from_array(image_array)
        image.write_to_file(str(path))
    except Exception as e:
        print(f"Error saving image {path}: {e}")


# --- Main execution block ---
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # --- Parameters for optimized inference ---
    # The original script hardcoded batch_size=1 and num_threads=0.
    # These have been removed to allow configuration via command-line arguments for better performance.
    # Example: --batch_size 16 --num_threads 8 --epoch 15

    # These options are generally good for deterministic testing
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True  # no flip during testing

    # These are for standard inference runs
    opt.display_id = -1  # no visdom display
    opt.phase = 'infer_valid_test'  # Set a custom phase name if needed

    # Create dataset and model
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)  # Loads networks, sets up schedulers

    # Get the actual configuration from options to provide user feedback
    batch_size = opt.batch_size
    num_threads = opt.num_threads

    print("--- Inference Configuration ---")
    print(f"Batch Size: {batch_size}")
    print(f"Dataloader Threads: {num_threads}")
    print(f"Loading from epoch: {opt.epoch}")
    print("-----------------------------")

    # Create results directory
    save_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    os.makedirs(save_dir, exist_ok=True)

    # Set model to evaluation mode. This is crucial for correct results.
    model.eval()

    # --- Main inference and saving loop ---
    # Create a thread pool executor that will live for the duration of the loop.
    # A value between 8 and 16 is a good starting point for max_workers for saving images.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(8, num_threads)) as executor:
        for data in tqdm(dataset, desc="Processing batches"):
            model.set_input(data)  # Unpack data for the entire batch
            model.infer()  # Run inference on the entire batch

            # Prepare arguments for each save task
            # .unbind(0) efficiently splits the batch tensor into a tuple of single-image tensors
            tensors_to_save = list(model.fake_B_denorm)

            # Create a full save path for each image in the batch
            paths_to_save = [Path(save_dir) / (Path(p).stem + '.tiff') for p in data['A_paths']]

            # Submit all save tasks for the current batch to the thread pool.
            # The .map function will handle distributing the tasks to worker threads.
            executor.map(save_image, tensors_to_save, paths_to_save)

    print(f"\nInference complete. Results saved in {save_dir}")