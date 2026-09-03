"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # Get total iterations to train
    n_iters = getattr(opt, 'n_iters', 100000)
    n_iters_decay = getattr(opt, 'n_iters_decay', 0)
    total_train_iters = n_iters + n_iters_decay
    
    print('Training for %d iterations (constant LR for %d iters, decay for %d iters)' % 
          (total_train_iters, n_iters, n_iters_decay))
    
    iter_data_time = time.time()    # timer for data loading per iteration
    dataset_iter = iter(dataset)    # create iterator for dataset
    
    while total_iters < total_train_iters:
        iter_start_time = time.time()  # timer for computation per iteration
        
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        # Update learning rate based on current iteration
        model.update_learning_rate(current_iter=total_iters)
        
        # Get next batch of data
        try:
            data = next(dataset_iter)
        except StopIteration:
            # Restart dataset iterator when exhausted
            dataset_iter = iter(dataset)
            visualizer.reset()  # reset the visualizer
            data = next(dataset_iter)

        total_iters += 1  # increment iteration counter

        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            # visualizer.display_current_results(model.get_current_visuals(), total_iters, save_result)

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            # Use pseudo epoch and iteration for compatibility with visualizer
            pseudo_epoch = total_iters // dataset_size if dataset_size > 0 else total_iters // 1000
            pseudo_iter = total_iters % dataset_size if dataset_size > 0 else total_iters % 1000
            visualizer.print_current_losses(pseudo_epoch, pseudo_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                counter_ratio = float(pseudo_iter) / dataset_size if dataset_size > 0 else 0.0
                visualizer.plot_current_losses(pseudo_epoch, counter_ratio, losses)

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (total_iters %d)' % total_iters)
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        # Save checkpoint at regular intervals (equivalent to save_epoch_freq)
        if opt.save_epoch_freq > 0 and total_iters % (opt.save_epoch_freq * dataset_size) == 0:
            print('saving the model at iteration %d' % total_iters)
            model.save_networks('latest')
            model.save_networks('iter_%d' % total_iters)

        iter_data_time = time.time()
        
        if total_iters >= total_train_iters:
            print('Reached %d iterations. Stopping training and saving model.' % total_train_iters)
            print('saving the model at the end of training, iters %d' % total_iters)
            model.save_networks('latest')
            model.save_networks('iter_%d' % total_iters)
            break

    print('Training finished at iteration %d' % total_iters)
