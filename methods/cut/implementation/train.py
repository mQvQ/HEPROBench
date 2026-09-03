import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.

    model = create_model(opt)  # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    
    optimize_time = 0.1
    times = []

    # Training loop based on iterations
    total_iterations = opt.total_iterations
    start_iteration = opt.continue_iteration if opt.continue_iteration > 0 else 0
    
    # Create iterator for dataset to handle cycling
    dataset_iter = iter(dataset)
    
    # Initialize model on first iteration
    model_initialized = False
    
    iter_data_time = time.time()
    
    for iteration in range(start_iteration, total_iterations):
        iter_start_time = time.time()
        
        # Get next batch, restart iterator if exhausted
        try:
            data = next(dataset_iter)
        except StopIteration:
            dataset_iter = iter(dataset)
            data = next(dataset_iter)
        
        batch_size = data["A"].size(0)
        
        if iteration % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()
        optimize_start_time = time.time()
        
        # Initialize model on first iteration
        if not model_initialized:
            model.data_dependent_initialize(data)
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            model.parallelize()
            model_initialized = True
        
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
        
        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()
        optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

        if iteration % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
            save_result = iteration % opt.update_html_freq == 0
            model.compute_visuals()
            # visualizer.display_current_results(model.get_current_visuals(), iteration, save_result)

        if iteration % opt.print_freq == 0:  # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            # Use iteration for both epoch and iters parameters (for compatibility with visualizer)
            visualizer.print_current_losses(iteration, iteration, losses, optimize_time, t_data)
            if opt.display_id is None or opt.display_id > 0:
                progress = float(iteration) / total_iterations
                visualizer.plot_current_losses(iteration, progress, losses)

        if iteration % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (iteration %d)' % iteration)
            print(opt.name)  # it's useful to occasionally show the experiment name on console
            save_suffix = 'iter_%d' % iteration if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        # Save checkpoint at specified intervals
        if iteration > 0 and iteration % opt.save_per_iteration == 0:
            print('saving the model at iteration %d' % iteration)
            model.save_networks('latest')
            model.save_networks('iter_%d' % iteration)

        # Update learning rate (skip first iteration to ensure model is initialized)
        if iteration > 0 and iteration % opt.lr_update_freq == 0:
            model.update_learning_rate(iteration=iteration)

        iter_data_time = time.time()
    
    # Save final checkpoint
    print('Training completed. Saving final checkpoint at iteration %d...' % total_iterations)
    model.save_networks('latest')
    model.save_networks('iter_%d' % total_iterations)
    print('Training finished.')
