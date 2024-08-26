import torch
from .base_model import BaseModel
from . import networks


class SimpleModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.ax_coef = opt.ax_coef
        self.cor_coef = opt.cor_coef
        self.ps = opt.patch_size

        self.loss_names = ['G_GAN_cor', 'G_GAN_ax', 'G_L1_cor', 'G_L1_ax', 'D_cor_real', 'D_cor_fake', 'D_ax_real',
                               'D_ax_fake']

        self.visual_names = ['real_A', 'fake_B_cor', 'real_B_cor', 'fake_B_ax', 'real_B_ax']

        if self.isTrain:
            self.model_names = ['G', 'D_cor', 'D_ax']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD_cor = networks.define_D(opt.input_nc + opt.input_nc, opt.ndf_cor, opt.netD_cor,
                                              opt.n_layers_D_cor, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_ax = networks.define_D(opt.input_nc + opt.input_nc, opt.ndf_ax, opt.netD_ax,
                                             opt.n_layers_D_ax, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_cor = torch.optim.Adam(self.netD_cor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_ax = torch.optim.Adam(self.netD_ax.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_cor)
            self.optimizers.append(self.optimizer_D_ax)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A_cor = input['A' if AtoB else 'B'].float().to(self.device)

        if self.isTrain:
            self.real_B_cor = input['B' if AtoB else 'A'].float().to(self.device)
            self.real_C_cor = input['C'].float().to(self.device)
            self.real_B_ax = torch.movedim(self.real_C_cor, (0, 1, 2, 3, 4), (0, 1, 3, 2, 4))

    def forward(self):
        self.fake_B_cor = self.netG(self.real_A_cor)
        self.fake_B_ax = torch.movedim(self.fake_B_cor, (0, 1, 2, 3, 4), (0, 1, 3, 2, 4))
        self.real_A_ax = torch.movedim(self.real_A_cor, (0, 1, 2, 3, 4), (0, 1, 3, 2, 4))


    def backward_D_cor(self):
        real_A_cor = torch.permute(self.real_A_cor, (0, 2, 1, 3, 4))
        real_A_cor = real_A_cor.reshape(-1, 1, self.ps, self.ps)

        fake_B_cor = torch.permute(self.fake_B_cor, (0, 2, 1, 3, 4))
        fake_B_cor = fake_B_cor.reshape(-1, 1, self.ps, self.ps)

        real_B_cor = self.real_B_cor.view(-1, 1, self.ps, self.ps)

        # Fake
        fake_AB_cor = torch.cat((real_A_cor, fake_B_cor), 1)
        pred_fake_cor = self.netD_cor(fake_AB_cor.detach())

        self.loss_D_cor_fake = self.criterionGAN(pred_fake_cor, False)

        # Real
        real_AB_cor = torch.cat((real_A_cor, real_B_cor), 1)
        pred_real_cor = self.netD_cor(real_AB_cor)

        self.loss_D_cor_real = self.criterionGAN(pred_real_cor, True)

        # total D Coronal loss
        self.loss_D_cor = (self.loss_D_cor_fake + self.loss_D_cor_real) * 0.5
        self.loss_D_cor.backward()


    def backward_D_ax(self):
        fake_B_ax = torch.permute(self.fake_B_ax, (0, 2, 1, 3, 4))
        fake_B_ax = fake_B_ax.reshape(-1, 1, self.ps, self.ps)

        real_A_ax = torch.permute(self.real_A_ax, (0, 2, 1, 3, 4))
        real_A_ax = real_A_ax.reshape(-1, 1, self.ps, self.ps)

        real_B_ax = torch.permute(self.real_B_ax, (0, 2, 1, 3, 4))
        real_B_ax = real_B_ax.reshape(-1, 1, self.ps, self.ps)

        # Fake
        fake_AB_ax = torch.cat((real_A_ax, fake_B_ax), 1)
        pred_fake_ax = self.netD_ax(fake_AB_ax.detach())

        self.loss_D_ax_fake = self.criterionGAN(pred_fake_ax, False)

        # Real
        real_AB_ax = torch.cat((real_A_ax, real_B_ax), 1)
        pred_real_ax = self.netD_ax(real_AB_ax)

        self.loss_D_ax_real = self.criterionGAN(pred_real_ax, True)

        # total D Axial loss
        self.loss_D_ax = (self.loss_D_ax_fake + self.loss_D_ax_real) * 0.5
        self.loss_D_ax.backward()

    def backward_G(self):
        # Coronal
        real_A_cor = torch.permute(self.real_A_cor, (0, 2, 1, 3, 4))
        real_A_cor = real_A_cor.reshape(-1, 1, self.ps, self.ps)

        fake_B_cor = torch.permute(self.fake_B_cor, (0, 2, 1, 3, 4))
        fake_B_cor = fake_B_cor.reshape(-1, 1, self.ps, self.ps)

        real_B_cor = self.real_B_cor.view(-1, 1, self.ps, self.ps)

        fake_AB_cor = torch.cat((real_A_cor, fake_B_cor), 1)
        pred_fake_cor = self.netD_cor(fake_AB_cor)

        self.loss_G_GAN_cor = self.criterionGAN(pred_fake_cor, True)
        self.loss_G_L1_cor = self.criterionL1(fake_B_cor, real_B_cor) * self.opt.lambda_L1

        # Axial
        fake_B_ax = torch.permute(self.fake_B_ax, (0, 2, 1, 3, 4))
        fake_B_ax = fake_B_ax.reshape(-1, 1, self.ps, self.ps)

        real_A_ax = torch.permute(self.real_A_ax, (0, 2, 1, 3, 4))
        real_A_ax = real_A_ax.reshape(-1, 1, self.ps, self.ps)

        real_B_ax = torch.permute(self.real_B_ax, (0, 2, 1, 3, 4))
        real_B_ax = real_B_ax.reshape(-1, 1, self.ps, self.ps)

        fake_AB_ax = torch.cat((real_A_ax, fake_B_ax), 1)
        pred_fake_ax = self.netD_ax(fake_AB_ax)

        self.loss_G_GAN_ax = self.criterionGAN(pred_fake_ax, True)
        self.loss_G_L1_ax = self.criterionL1(fake_B_ax, real_B_ax) * self.opt.lambda_L1

        # total G loss
        self.loss_G = self.cor_coef * (self.loss_G_GAN_cor + self.loss_G_L1_cor) + self.ax_coef * (
                    self.loss_G_GAN_ax + self.loss_G_L1_ax)

        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()

        # update D Coronal
        self.set_requires_grad(self.netD_cor, True)
        self.optimizer_D_cor.zero_grad()
        self.backward_D_cor()
        self.optimizer_D_cor.step()

        # update D Axial
        self.set_requires_grad(self.netD_ax, True)
        self.optimizer_D_ax.zero_grad()
        self.backward_D_ax()
        self.optimizer_D_ax.step()

        # update G
        self.set_requires_grad(self.netD_cor, False)
        self.set_requires_grad(self.netD_ax, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()