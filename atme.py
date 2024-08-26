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

See options/base_atme_options.py and options/train_atme_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import random
import time
import torch
import numpy as np
from data import create_atme_train_dataset, create_atme_test_dataset
from models import create_model
from util.visualizer import Visualizer, save_atme_images
from options.train_atme_options import TrainAtmeOptions


torch.manual_seed(13)
random.seed(13)
np.random.seed(13)

def train(opt):
    opt.isTrain = True
    opt.save_dir = os.path.join(opt.main_root, opt.atme_root, opt.exp_name)
    opt.data_dir = os.path.join(opt.main_root, opt.atme_root, opt.data_name)
    save_fig_dir = os.path.join(opt.save_dir, 'train_figures')

    dataset = create_atme_train_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt, dataset)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0


    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            if 'n_save_noisy' in vars(opt):
                model.set_input(data, epoch)
            else:
                model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                visuals = model.get_current_visuals()
                slice_num = i if opt.batch_size == 1 else random.randint(0, opt.batch_size)
                save_atme_images(visuals, save_fig_dir, slice_num, iter_num=total_iters, epoch=epoch)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Save D_real and D_fake
        visualizer.save_D_losses(model.get_current_losses())

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

def test(opt):
    opt.isTrain = False

    opt.save_dir = os.path.join(opt.main_root, opt.atme_root, opt.exp_name)
    opt.data_dir = os.path.join(opt.main_root, opt.atme_root, opt.data_name)

    save_fig_dir = os.path.join(opt.save_dir, 'test_figures')

    cases_paths = torch.load(os.path.join(opt.main_root, f'coronal_cases_paths.pt'))

    plot_slice = 100 if opt.plane == 'coronal' else 250

    for i, cor_case in enumerate(cases_paths):

        save_dir = os.path.join(opt.data_dir, 'generation', f'case_{i}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        dataset = create_atme_test_dataset(opt, cor_case, i)
        model = create_model(opt, dataset)
        model.setup(opt)
        model.eval()

        gen_vol = torch.zeros((len(dataset), 512, 512))

        for j, data in enumerate(dataset):
            model.set_input(data)
            model.test()
            gen_vol[j, :, :] = model.fake_B[0, 0, :, :]

            if j == plot_slice:
                visuals = model.get_current_visuals()
                save_atme_images(visuals, save_fig_dir, j, case_num=i)

        if opt.plane == 'axial':
            gen_vol = dataset.dataset.crop_volume(gen_vol)

        torch.save(gen_vol, os.path.join(save_dir, 'atme_vol.pt'))


if __name__ == '__main__':
    atme_opt = TrainAtmeOptions().parse()
    if atme_opt.isTrain:
        train(atme_opt)
    else:
        test(atme_opt)
