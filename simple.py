import os
import time
import torch
import random
import numpy as np
import pandas as pd
from data import create_simple_train_dataset, create_simple_test_dataset
from data.preprocess import reconstruct_volume, pad_volume, find_grayscale_limits, save_nifti
from models import create_model
from util.visualizer import Visualizer, plot_simple_train_results, plot_simple_test_results
from util.util import mkdir
from options.simple_options import SimpleOptions


torch.manual_seed(13)
random.seed(13)
np.random.seed(13)


def train(opt):
    opt.isTrain = True

    opt.save_dir = os.path.join(opt.main_root, opt.model_root, opt.exp_name)
    mkdir(opt.save_dir)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    train_loader = create_simple_train_dataset(opt)
    print('prepare data_loader done')

    total_iters = 0

    figures_path = os.path.join(opt.save_dir, 'figures', 'train')
    mkdir(figures_path)

    slice_index = int(opt.patch_size / 2)

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

            if i == 0 and epoch % 5 == 0:
                plot_simple_train_results(model, epoch, figures_path, opt.planes, slice_index)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

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

    opt.save_dir = os.path.join(opt.main_root, opt.model_root, opt.exp_name)
    opt.data_dir = os.path.join(opt.main_root, opt.model_root, opt.data_name)
    figures_path = os.path.join(opt.save_dir, 'figures', 'test')
    mkdir(figures_path)

    model = create_model(opt)
    model.setup(opt)

    df = pd.read_csv(os.path.join(opt.csv_name), low_memory=False)
    cases_paths = df.loc[:, opt.eval_plane]

    if opt.global_min == 0 and opt.global_max == 0:
        opt.global_min, opt.global_max = find_grayscale_limits(cases_paths, opt.data_format)

    for case_idx, case in enumerate(cases_paths):
        print(f'case no: {case_idx} / {len(cases_paths)}, {case=}')

        data_loader = create_simple_test_dataset(case, opt)

        output_patches_3d = []
        for l, data in enumerate(data_loader):
            model.set_requires_grad(model.netG, False)
            model.set_input(data)
            model.forward()
            if opt.eval_plane == 'coronal':
                output_patches_3d.append(model.fake_B_cor)
            elif opt.eval_plane == 'axial':
                output_patches_3d.append(model.fake_B_ax)
            elif opt.eval_plane == 'sagittal':
                output_patches_3d.append(model.fake_B_sag)

        DS = data_loader.dataset
        interp_vol = DS.padded_case

        recon_vol = reconstruct_volume(opt, output_patches_3d, interp_vol.shape)

        interp_vol = pad_volume(interp_vol.squeeze(), opt.vol_cube_dim)
        recon_vol = pad_volume(recon_vol.squeeze(), opt.vol_cube_dim)

        plot_simple_test_results(interp_vol, recon_vol, figures_path, case_idx, opt)

        save_dir = os.path.join(opt.data_dir, 'test', f'case_{case_idx}')
        mkdir(save_dir)



        torch.save(recon_vol.cpu().detach(), os.path.join(save_dir, f'simple_vol.pt'))
        torch.save(interp_vol.cpu().detach(), os.path.join(save_dir, 'interp_vol.pt'))

        if opt.save_nifti:
            save_nifti(recon_vol, os.path.join(save_dir, f'simple_vol.nii.gz'))
            save_nifti(interp_vol, os.path.join(save_dir, 'interp_vol.nii.gz'))

if __name__ == '__main__':
    simple_opt = SimpleOptions().parse()
    if simple_opt.isTrain:
        train(simple_opt)
    else:
        test(simple_opt)