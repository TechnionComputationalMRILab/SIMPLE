import os
import time
import torch
import random
import numpy as np
from data import create_simple_train_dataset, create_simple_test_dataset
from data.preprocess import reconstruct_volume, pad_volume
from models import create_model
from util.visualizer import Visualizer, plot_simple_train_results, plot_simple_test_results
from util.util import mkdir, mkdirs
from options.train_simple_options import TrainSimpleOptions


torch.manual_seed(13)
random.seed(13)
np.random.seed(13)

def setup(opt):
    mkdir(opt.main_root)
    mkdir(os.path.join(opt.main_root, opt.simple_root, opt.exp_name))
    mkdir(os.path.join(opt.main_root, opt.atme_cor_root, opt.exp_name))
    mkdir(os.path.join(opt.main_root, opt.atme_ax_root, opt.exp_name))


def train(opt):
    opt.save_dir = os.path.join(opt.main_root, opt.simple_root, opt.exp_name)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    train_loader = create_simple_train_dataset(opt)
    print('prepare data_loader done')

    dataset_size = len(train_loader)
    print('The number of training images = %d' % dataset_size)

    total_iters = 0

    figures_path = os.path.join(opt.save_dir, 'figures', 'train')
    mkdir(figures_path)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        if epoch > 1:
            model.update_learning_rate()
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if i == 0 and epoch % 10 == 0:
                plot_simple_train_results(model, epoch, figures_path)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        losses = model.get_current_losses()
        visualizer.save_to_tensorboard_writer(epoch, losses)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

def test(opt):
    opt.isTrain = False
    opt.save_dir = os.path.join(opt.main_root, opt.simple_root, opt.exp_name)
    opt.data_dir = os.path.join(opt.main_root, opt.simple_root, opt.data_name)
    figures_path = os.path.join(opt.save_dir, 'figures', 'test')
    mkdir(figures_path)

    model = create_model(opt)
    model.setup(opt)

    # visualizer = Visualizer(opt) #TODO: to add a plot of simple results


    cor_cases_paths = torch.load(os.path.join(opt.main_root, f'coronal_cases_paths.pt'))

    for case_idx, cor_case in enumerate(cor_cases_paths):
        print(f'{case_idx=}')
        data_loader = create_simple_test_dataset(cor_case, opt)

        output_patches_3d = []
        for l, data in enumerate(data_loader):
            model.set_requires_grad(model.netG, False)
            model.set_input(data)
            model.forward()
            output_patches_3d.append(model.fake_B_cor)

        DS = data_loader.dataset
        interp_vol = DS.padded_case
        recon_vol = reconstruct_volume(opt, output_patches_3d, interp_vol.shape)

        interp_vol = pad_volume(interp_vol.squeeze())
        recon_vol = pad_volume(recon_vol.squeeze())

        plot_simple_test_results(interp_vol, recon_vol, figures_path, case_idx)

        save_dir = os.path.join(opt.data_dir, 'test', f'case_{case_idx}')
        mkdir(save_dir)

        torch.save(recon_vol.cpu().detach(), os.path.join(save_dir, 'simple_vol.pt'))


if __name__ == '__main__':
    simple_opt = TrainSimpleOptions().parse()
    if simple_opt.isTrain:
        train(simple_opt)
    else:
        test(simple_opt)