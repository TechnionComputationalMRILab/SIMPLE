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

        self.ps = opt.patch_size

        self.loss_names = []
        self.visual_names = ['real_A']
        self.model_names = ['G']

        for plane in opt.planes:
            if plane == 'coronal' : suffix = 'cor'
            if plane == 'sagittal': suffix = 'sag'
            if plane == 'axial'   : suffix = 'ax'
            self.loss_names += [f'G_GAN_{suffix}', f'G_L1_{suffix}', f'D_{suffix}_real', f'D_{suffix}_fake']
            self.visual_names += [f'fake_B_{suffix}', f'real_B_{suffix}']
            if self.isTrain:
                self.model_names += [f'D_{suffix}']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if 'coronal' in opt.planes:
                self.netD_cor = networks.define_D(opt.input_nc + opt.input_nc, opt.ndf_cor, opt.netD_cor,
                                                  opt.n_layers_D_cor, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if 'axial' in opt.planes:
                self.netD_ax = networks.define_D(opt.input_nc + opt.input_nc, opt.ndf_ax, opt.netD_ax,
                                                 opt.n_layers_D_ax, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if 'sagittal' in opt.planes:
                self.netD_sag = networks.define_D(opt.input_nc + opt.input_nc, opt.ndf_sag, opt.netD_sag,
                                                 opt.n_layers_D_sag, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if 'coronal' in opt.planes:
                self.optimizer_D_cor = torch.optim.Adam(self.netD_cor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_cor)
            if 'axial' in opt.planes:
                self.optimizer_D_ax = torch.optim.Adam(self.netD_ax.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_ax)
            if 'sagittal' in opt.planes:
                self.optimizer_D_sag = torch.optim.Adam(self.netD_sag.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_sag)

    def set_input(self, input):

        if self.opt.eval_plane == 'coronal':
            self.real_A_cor = input['Interpolation'].float().to(self.device)
        elif self.opt.eval_plane == 'axial':
            self.real_A_ax = input['Interpolation'].float().to(self.device)
        elif self.opt.eval_plane == 'sagittal':
            self.real_A_sag = input['Interpolation'].float().to(self.device)

        if self.isTrain:
            if self.opt.eval_plane == 'coronal':
                self.real_B_cor = input['Coronal'].float().to(self.device)
                if 'axial' in self.opt.planes:
                    self.real_B_ax = torch.movedim(input['Axial'].float().to(self.device), (0, 1, 2, 3, 4), (0, 1, 3, 2, 4))
                if 'sagittal' in self.opt.planes:
                    self.real_B_sag = torch.movedim(input['Sagittal'].float().to(self.device), (0, 1, 2, 3, 4), (0, 1, 4, 3, 2))


            elif self.opt.eval_plane == 'axial':
                self.real_B_ax = input['Axial'].float().to(self.device)
                if 'coronal' in self.opt.planes:
                    self.real_B_cor = torch.movedim(input['Coronal'].float().to(self.device), (0, 1, 2, 3, 4), (0, 1, 3, 2, 4))
                if 'sagittal' in self.opt.planes:
                    self.real_B_sag = torch.movedim(input['Sagittal'].float().to(self.device), (0, 1, 2, 3, 4), (0, 1, 3, 4, 2))

            elif self.opt.eval_plane == 'sagittal':
                self.real_B_sag = input['Sagittal'].float().to(self.device)
                if 'coronal' in self.opt.planes:
                    self.real_B_cor = torch.movedim(input['Coronal'].float().to(self.device), (0, 1, 2, 3, 4), (0, 1, 4, 3, 2))
                if 'axial' in self.opt.planes:
                    self.real_B_ax = torch.movedim(input['Axial'].float().to(self.device), (0, 1, 2, 3, 4), (0, 1, 4, 2, 3))



    def forward(self):
        if self.opt.eval_plane == 'coronal':
            self.fake_B_cor = self.netG(self.real_A_cor)
            if 'axial' in self.opt.planes:
                self.fake_B_ax = torch.movedim(self.fake_B_cor, (0, 1, 2, 3, 4), (0, 1, 3, 2, 4))
                self.real_A_ax = torch.movedim(self.real_A_cor, (0, 1, 2, 3, 4), (0, 1, 3, 2, 4))
            if 'sagittal' in self.opt.planes:
                self.fake_B_sag = torch.movedim(self.fake_B_cor, (0, 1, 2, 3, 4), (0, 1, 4, 3, 2))
                self.real_A_sag = torch.movedim(self.real_A_cor, (0, 1, 2, 3, 4), (0, 1, 4, 3, 2))

        elif self.opt.eval_plane == 'axial':
            self.fake_B_ax = self.netG(self.real_A_ax)
            if 'coronal' in self.opt.planes:
                self.fake_B_cor = torch.movedim(self.fake_B_ax, (0, 1, 2, 3, 4), (0, 1, 3, 2, 4))
                self.real_A_cor = torch.movedim(self.real_A_ax, (0, 1, 2, 3, 4), (0, 1, 3, 2, 4))
            if 'sagittal' in self.opt.planes:
                self.fake_B_sag = torch.movedim(self.fake_B_ax, (0, 1, 2, 3, 4), (0, 1, 3, 4, 2))
                self.real_A_sag = torch.movedim(self.real_A_ax, (0, 1, 2, 3, 4), (0, 1, 3, 4, 2))

        elif self.opt.eval_plane == 'sagittal':
            self.fake_B_sag = self.netG(self.real_A_sag)
            if 'coronal' in self.opt.planes:
                self.fake_B_cor = torch.movedim(self.fake_B_sag, (0, 1, 2, 3, 4), (0, 1, 4, 3, 2))
                self.real_A_cor = torch.movedim(self.real_A_sag, (0, 1, 2, 3, 4), (0, 1, 4, 3, 2))
            if 'axial' in self.opt.planes:
                self.fake_B_ax = torch.movedim(self.fake_B_sag, (0, 1, 2, 3, 4), (0, 1, 4, 2, 3))
                self.real_A_ax = torch.movedim(self.real_A_sag, (0, 1, 2, 3, 4), (0, 1, 4, 2, 3))

    def backward_D_cor(self):
        real_A_cor = torch.permute(self.real_A_cor, (0, 2, 1, 3, 4))
        real_A_cor = real_A_cor.reshape(-1, 1, self.ps, self.ps)

        fake_B_cor = torch.permute(self.fake_B_cor, (0, 2, 1, 3, 4))
        fake_B_cor = fake_B_cor.reshape(-1, 1, self.ps, self.ps)

        real_B_cor = torch.permute(self.real_B_cor, (0, 2, 1, 3, 4))
        real_B_cor = real_B_cor.reshape(-1, 1, self.ps, self.ps)

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

    def backward_D_sag(self):
        fake_B_sag = torch.permute(self.fake_B_sag, (0, 2, 1, 3, 4))
        fake_B_sag = fake_B_sag.reshape(-1, 1, self.ps, self.ps)

        real_A_sag = torch.permute(self.real_A_sag, (0, 2, 1, 3, 4))
        real_A_sag = real_A_sag.reshape(-1, 1, self.ps, self.ps)

        real_B_sag = torch.permute(self.real_B_sag, (0, 2, 1, 3, 4))
        real_B_sag = real_B_sag.reshape(-1, 1, self.ps, self.ps)

        # Fake
        fake_AB_sag = torch.cat((real_A_sag, fake_B_sag), 1)
        pred_fake_sag = self.netD_sag(fake_AB_sag.detach())

        self.loss_D_sag_fake = self.criterionGAN(pred_fake_sag, False)

        # Real
        real_AB_sag = torch.cat((real_A_sag, real_B_sag), 1)
        pred_real_sag = self.netD_sag(real_AB_sag)

        self.loss_D_sag_real = self.criterionGAN(pred_real_sag, True)

        # total D Sagittal loss
        self.loss_D_sag = (self.loss_D_sag_fake + self.loss_D_sag_real) * 0.5
        self.loss_D_sag.backward()

    def backward_G(self):
        # Coronal
        real_A_cor = torch.permute(self.real_A_cor, (0, 2, 1, 3, 4))
        real_A_cor = real_A_cor.reshape(-1, 1, self.ps, self.ps)

        fake_B_cor = torch.permute(self.fake_B_cor, (0, 2, 1, 3, 4))
        fake_B_cor = fake_B_cor.reshape(-1, 1, self.ps, self.ps)

        real_B_cor = torch.permute(self.real_B_cor, (0, 2, 1, 3, 4))
        real_B_cor = real_B_cor.reshape(-1, 1, self.ps, self.ps)

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

        # Sagittal
        fake_B_sag = torch.permute(self.fake_B_sag, (0, 2, 1, 3, 4))
        fake_B_sag = fake_B_sag.reshape(-1, 1, self.ps, self.ps)

        real_A_sag = torch.permute(self.real_A_sag, (0, 2, 1, 3, 4))
        real_A_sag = real_A_sag.reshape(-1, 1, self.ps, self.ps)

        real_B_sag = torch.permute(self.real_B_sag, (0, 2, 1, 3, 4))
        real_B_sag = real_B_sag.reshape(-1, 1, self.ps, self.ps)

        fake_AB_sag = torch.cat((real_A_sag, fake_B_sag), 1)
        pred_fake_sag = self.netD_sag(fake_AB_sag)

        self.loss_G_GAN_sag = self.criterionGAN(pred_fake_sag, True)
        self.loss_G_L1_sag = self.criterionL1(fake_B_sag, real_B_sag) * self.opt.lambda_L1

        # total G loss
        self.loss_G = self.opt.cor_coef * (self.loss_G_GAN_cor + self.loss_G_L1_cor) + \
                      self.opt.ax_coef * (self.loss_G_GAN_ax + self.loss_G_L1_ax) + \
                      self.opt.sag_coef * (self.loss_G_GAN_sag + self.loss_G_L1_sag)

        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()

        # update D Coronal
        if 'coronal' in self.opt.planes:
            self.set_requires_grad(self.netD_cor, True)
            self.optimizer_D_cor.zero_grad()
            self.backward_D_cor()
            self.optimizer_D_cor.step()

        # update D Axial
        if 'axial' in self.opt.planes:
            self.set_requires_grad(self.netD_ax, True)
            self.optimizer_D_ax.zero_grad()
            self.backward_D_ax()
            self.optimizer_D_ax.step()

        if 'sagittal' in self.opt.planes:
            self.set_requires_grad(self.netD_sag, True)
            self.optimizer_D_sag.zero_grad()
            self.backward_D_sag()
            self.optimizer_D_sag.step()

        # update G
        if 'coronal'  in self.opt.planes: self.set_requires_grad(self.netD_cor, False)
        if 'axial'    in self.opt.planes: self.set_requires_grad(self.netD_ax, False)
        if 'sagittal' in self.opt.planes: self.set_requires_grad(self.netD_sag, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()