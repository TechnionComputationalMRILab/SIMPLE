import numpy as np
import os
import sys
import time
import torch
from . import util, html
from subprocess import Popen, PIPE
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_atme_images(visuals, results_dir, slice_num, case_num=None, iter_num=None, epoch=None):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    fig, axs = plt.subplots(1, len(visuals.items()))
    j = 0
    for label, im_data in visuals.items():
        img = im_data[0, 0].cpu().detach().numpy() if (len(im_data.shape) == 4) else im_data[0].cpu().detach().numpy()
        axs[j].imshow(img, cmap="gray")
        axs[j].axis("off")
        axs[j].set_title(f'{label}')
        j += 1

    fig_name = f'case_{case_num}_slice_{slice_num}.pdf' if iter_num == None else f'epoch_{epoch}_iter_{iter_num}_slice_{slice_num}.pdf'
    img_pdf_path = os.path.join(results_dir, fig_name)
    plt.savefig(img_pdf_path)
    plt.close()

def plot_simple_train_results(model, epoch, figures_path, planes, slice_index):
    print('-------------------PLOT SIMPLE TRAINING RESULTS----------------------')

    fig, axs = plt.subplots(len(planes), 3)

    for i, plane in enumerate(planes):
        if plane == 'coronal':
            interp_img = model.real_A_cor[0, 0, slice_index, :, :].squeeze().cpu().detach().numpy()
            atme_img = model.real_B_cor[0, 0, slice_index, :, :].cpu().detach().numpy()
            simple_img = model.fake_B_cor[0, 0, slice_index, :, :].squeeze().cpu().detach().numpy()
        elif plane == 'axial':
            interp_img = model.real_A_ax[0, 0, slice_index, :, :].squeeze().cpu().detach().numpy()
            atme_img = model.real_B_ax[0, 0, slice_index, :, :].cpu().detach().numpy()
            simple_img = model.fake_B_ax[0, 0, slice_index, :, :].squeeze().cpu().detach().numpy()
        elif plane == 'sagittal':
            interp_img = model.real_A_sag[0, 0, slice_index, :, :].squeeze().cpu().detach().numpy()
            atme_img = model.real_B_sag[0, 0, slice_index, :, :].cpu().detach().numpy()
            simple_img = model.fake_B_sag[0, 0, slice_index, :, :].squeeze().cpu().detach().numpy()

        axs[i, 0].imshow(interp_img, vmin=-1, vmax=1, cmap="gray")
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 0].set_ylabel(f'{plane}', fontsize="20")
        axs[i, 0].set_title(f'Interpolation')

        axs[i, 1].imshow(atme_img, vmin=-1, vmax=1, cmap="gray")
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])
        axs[i, 1].set_title(f'ATME')

        axs[i, 2].imshow(simple_img, vmin=-1, vmax=1, cmap="gray")
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])
        axs[i, 2].set_title(f'SIMPLE')

    fig.align_ylabels()
    plt.savefig(os.path.join(figures_path, f'results_epoch_{epoch}.png'))
    plt.close()

def plot_simple_test_results(interp_vol, simple_vol, figures_path, case_idx, opt):
    fig, axs = plt.subplots(len(opt.planes), 2)

    for i, plane in enumerate(opt.planes):
        if opt.eval_plane == 'coronal':
            if plane == 'coronal':
                interp_slice = interp_vol[150, :, :].cpu().detach().numpy()
                simple_slice = simple_vol[150, :, :].cpu().detach().numpy()
            elif plane == 'axial':
                interp_slice = torch.movedim(interp_vol, (0, 1, 2), (1, 0, 2))[150, :, :].cpu().detach().numpy()
                simple_slice = torch.movedim(simple_vol, (0, 1, 2), (1, 0, 2))[150, :, :].cpu().detach().numpy()
            elif plane == 'sagittal':
                interp_slice = torch.movedim(interp_vol, (0, 1, 2), (2, 1, 0))[150, :, :].cpu().detach().numpy()
                simple_slice = torch.movedim(simple_vol, (0, 1, 2), (2, 1, 0))[150, :, :].cpu().detach().numpy()

        elif opt.eval_plane == 'axial':
            if plane == 'coronal':
                interp_slice = torch.movedim(interp_vol, (0, 1, 2), (1, 0, 2))[150, :, :].cpu().detach().numpy()
                simple_slice = torch.movedim(simple_vol, (0, 1, 2), (1, 0, 2))[150, :, :].cpu().detach().numpy()
            elif plane == 'axial':
                interp_slice = interp_vol[150, :, :].cpu().detach().numpy()
                simple_slice = simple_vol[150, :, :].cpu().detach().numpy()
            elif plane == 'sagittal':
                interp_slice = torch.movedim(interp_vol, (0, 1, 2), (1, 2, 0))[150, :, :].cpu().detach().numpy()
                simple_slice = torch.movedim(simple_vol, (0, 1, 2), (1, 2, 0))[150, :, :].cpu().detach().numpy()

        elif opt.eval_plane == 'sagittal':
            if plane == 'coronal':
                interp_slice = torch.movedim(interp_vol, (0, 1, 2), (2, 1, 0))[100, :, :].cpu().detach().numpy()
                simple_slice = torch.movedim(simple_vol, (0, 1, 2), (2, 1, 0))[100, :, :].cpu().detach().numpy()
            elif plane == 'axial':
                interp_slice = torch.movedim(interp_vol, (0, 1, 2), (2, 0, 1))[100, :, :].cpu().detach().numpy()
                simple_slice = torch.movedim(simple_vol, (0, 1, 2), (2, 0, 1))[100, :, :].cpu().detach().numpy()
            elif plane == 'sagittal':
                interp_slice = interp_vol[150, :, :].cpu().detach().numpy()
                simple_slice = simple_vol[150, :, :].cpu().detach().numpy()

        axs[i, 0].imshow(interp_slice, vmin=-1, vmax=1, cmap="gray")
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        axs[i, 0].set_ylabel(f'{plane}', fontsize="10")
        axs[i, 0].set_title(f'Interpolation')

        axs[i, 1].imshow(simple_slice, vmin=-1, vmax=1, cmap="gray")
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])
        axs[i, 1].set_title(f'SIMPLE')

    fig.align_ylabels()
    plt.savefig(os.path.join(figures_path, f'case_{case_idx}_{opt.overlap_ratio}_gradual_{opt.overlap_ratio}_slice_100.png')) #_zero_0.33
    plt.close()

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.exp_name
        self.port = opt.display_port
        self.saved = False
        self.writer_dir = os.path.join(opt.save_dir, 'tensorboard')
        util.mkdirs([self.writer_dir])
        self.writer = SummaryWriter(self.writer_dir)

        # if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
        #     import visdom
        #     self.ncols = opt.display_ncols
        #     self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
        #     if not self.vis.check_connection():
        #         self.create_visdom_connections()

        # if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
        #     self.web_dir = os.path.join(opt.save_dir, opt.checkpoints_dir, 'web')
        #     self.img_dir = os.path.join(self.web_dir, 'images')
        #     print('create web directory %s...' % self.web_dir)
        #     util.mkdirs([self.web_dir, self.img_dir])

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.save_dir, opt.checkpoints_dir, 'loss_log.txt')
        self.log_name_csv = os.path.join(opt.save_dir, opt.checkpoints_dir, 'loss_log.csv')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            # webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            # for n in range(epoch, 0, -1):
            #     webpage.add_header('epoch [%d]' % n)
            #     ims, txts, links = [], [], []
            #
            #     for label, image_numpy in visuals.items():
            #         image_numpy = util.tensor2im(image)
            #         img_path = 'epoch%.3d_%s.png' % (n, label)
            #         ims.append(img_path)
            #         txts.append(label)
            #         links.append(img_path)
            #     webpage.add_images(ims, txts, links, width=self.win_size)
            # webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def save_D_losses(self, losses):
        line = f'{losses["D_real"]},{losses["D_fake"]}\n' if self.opt.dataset_mode == 'aligned' \
                  else f'{2*losses["D_A"]},{2*losses["D_B"]}\n'

        with open(self.log_name_csv, "a") as log_file:
            log_file.write(line) # save the discriminator losses

    def save_to_tensorboard_writer(self, epoch, losses):
        for k, v in losses.items():
            self.writer.add_scalar(k, v, epoch)