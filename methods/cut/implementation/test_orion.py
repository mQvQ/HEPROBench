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
def save_image(image_array, path, compression='lzw'):
    """
    Converts a single tensor to a numpy array and saves it as a TIFF image using pyvips.
    This function is designed to be called from multiple threads.
    """
    try:
        image = pyvips.Image.new_from_array(image_array)
        image.write_to_file(str(path))
    except Exception as e:
        print(f"Error saving image {path}: {e}")


# --- Main execution block ---
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    # --- Parameters for optimized inference ---
    # We remove the hardcoded batch_size=1 and num_threads=0 to allow for user configuration.
    # Set these via command line for best performance, e.g., --batch_size 16 --num_threads 8

    # These options are generally good for deterministic testing
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True  # no flip during testing

    # These are for standard inference runs
    opt.display_id = -1  # no visdom display
    opt.phase = 'infer_valid_test'
    opt.load_iter = False

    # Create dataset and model
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # Get the actual batch size and number of threads from options
    batch_size = opt.batch_size
    num_threads = opt.num_threads

    print(f"Starting inference with batch_size = {batch_size} and num_threads = {num_threads}")
    print(f"Results will be saved to: {opt.results_dir}")

    # Create results directory
    save_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    os.makedirs(save_dir, exist_ok=True)

    if opt.eval:
        model.eval()

    # --- Main inference and saving loop ---
    # Create a thread pool executor that will live for the duration of the loop.
    # We set max_workers based on num_threads for I/O tasks.
    # A value between 8 and 16 is usually a good starting point for saving images.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(8, num_threads)) as executor:
        for data in tqdm(dataset, desc="Processing batches"):
            model.set_input(data)  # Unpack data for the entire batch
            model.infer()  # Run inference on the entire batch

            # Prepare arguments for each save task
            # model.fake_B_denorm is a batch of images, e.g., tensor of shape [16, C, H, W]
            # data['A_paths'] is a list of 16 corresponding source paths

            # .unbind(0) efficiently splits the batch tensor into a tuple of single-image tensors
            tensors_to_save = list(model.fake_B_denorm)
            # print(len(tensors_to_save), tensors_to_save[0].shape)

            # Create a full save path for each image in the batch
            paths_to_save = [Path(save_dir) / (Path(p).stem + '.tiff') for p in data['A_paths']]

            # Submit all save tasks for the current batch to the thread pool.
            # The .map function will handle distributing the tasks to worker threads.
            # The main loop can continue to the next batch while the images are being saved in the background.
            executor.map(save_image, tensors_to_save, paths_to_save)

    print(f"\nInference complete. Results saved in {save_dir}")