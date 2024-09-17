import os
import torch
from .base_model import BaseModel
from . import networks
from util.image_pool import DiscPool
import util.util as util
from itertools import chain
# from data import create_dataset


class AtmeModel(BaseModel):
    """ This class implements the ATME model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet_256_attn' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    atme paper: https://arxiv.org/pdf/x.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For atme, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with instance norm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.add_argument('--n_save_noisy', type=int, default=0, help='number of D_t and W_t to keep track of')
        parser.add_argument('--mask_size', type=int, default=512)
        parser.add_argument('--dim', type=int, default=64, help='dim for the ddm UNet')
        parser.add_argument('--dim_mults', type=tuple, default=(1, 2, 4, 8), help='dim_mults for the ddm UNet')
        parser.add_argument('--groups', type=int, default=8, help='number of groups for GroupNorm within ResnetBlocks')
        parser.add_argument('--init_dim', type=int, default=64, help='output channels after initial conv2d of x_t')
        parser.add_argument('--learned_sinusoidal_cond', type=bool, default=False,
                            help='learn fourier features for positional embedding?')
        parser.add_argument('--random_fourier_features', type=bool, default=False,
                            help='random fourier features for positional embedding?')
        parser.add_argument('--learned_sinusoidal_dim', type=int, default=16,
                            help='twice the number of fourier frequencies to learn')
        parser.add_argument('--time_dim_mult', type=int, default=4,
                            help='dim * time_dim_mult amounts to output channels after time-MLP')

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser


    def __init__(self, opt, dataset):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        if self.isTrain:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
            self.model_names = ['G', 'D', 'W']
            opt.pool_size = 0
            opt.gan_mode = 'vanilla'
        else:
            self.visual_names = ['real_A', 'fake_B']
            self.model_names = ['G', 'W']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      **{'dim': opt.dim,
                                         'dim_mults': opt.dim_mults,
                                         'init_dim': opt.init_dim,
                                         'resnet_block_groups': opt.groups,
                                         'learned_sinusoidal_cond': opt.learned_sinusoidal_cond,
                                         'learned_sinusoidal_dim': opt.learned_sinusoidal_dim,
                                         'random_fourier_features': opt.random_fourier_features,
                                         'time_dim_mult': opt.time_dim_mult})

        self.netW = networks.define_W(opt.init_type, opt.init_gain, self.gpu_ids, output_size=opt.vol_cube_dim)
        self.disc_pool = DiscPool(opt, self.device, dataset, isTrain=self.isTrain)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(chain(self.netW.parameters(), self.netG.parameters()), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.save_noisy = True if opt.n_save_noisy > 0 else False
        if self.save_noisy:
            self.save_DW_idx = torch.randint(len(dataset), (opt.n_save_noisy,))
            self.img_DW_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images_noisy')
            util.mkdir(self.img_DW_dir)

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            self.print_networks(opt.verbose)
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix, pre_train_G_path=opt.pre_train_G_path, pre_train_W_path=opt.pre_train_W_path)

    def _save_DW(self, visuals):
        to_save = (self.batch_indices.view(1, -1) == self.save_DW_idx.view(-1, 1)).any(dim=0)
        if any(to_save) > 0:
            idx_to_save = torch.nonzero(to_save)[0]
            for label, images in visuals.items():
                for idx, image in zip(idx_to_save, images[to_save]):
                    img_idx = self.batch_indices[idx].item()
                    image_numpy = util.tensor2im(image[None])
                    img_path = os.path.join(self.img_DW_dir, f'epoch_{self.epoch:03d}_{label}_{img_idx}.png')
                    util.save_image(image_numpy, img_path)

    def set_input(self, input, epoch=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.epoch = epoch
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device, dtype=torch.float)
        if self.isTrain:
            self.real_B = input['B' if AtoB else 'A'].to(self.device, dtype=torch.float)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.batch_indices = input['batch_indices']
        self.disc_B = self.disc_pool.query(self.batch_indices)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.Disc_B = self.netW(self.disc_B)
        self.noisy_A = self.real_A * (1 + self.Disc_B)
        self.fake_B = self.netG(self.noisy_A, self.Disc_B)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        self.disc_B = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(self.disc_B, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # Save discriminator output
        self.disc_pool.insert(self.disc_B.detach(), self.batch_indices)
        if self.save_noisy:  # Save images corresponding to disc_B and Disc_B
            self._save_DW({'D': torch.sigmoid(self.disc_B), 'W': self.Disc_B})