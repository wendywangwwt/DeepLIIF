import torch
from .base_model import BaseModel
from . import networks


class SDGModel(BaseModel):
    """ This class implements the Synthetic Data Generation model, for learning a mapping from input images to modalities given paired data."""

    def __init__(self, opt):
        """Initialize the SDG class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # weights of the modalities in generating segmentation mask
        self.seg_weights = [1 / (self.opt.modalities_no + self.opt.input_no)] * (self.opt.modalities_no + self.opt.input_no) #[0.25, 0.15, 0.25, 0.1, 0.25]

        # loss weights in calculating the final loss
        self.loss_G_weights = [1 / (self.opt.modalities_no + self.opt.seg_no)] *  (self.opt.modalities_no + self.opt.seg_no) #[0.2, 0.2, 0.2, 0.2, 0.2]
        self.loss_D_weights = [1 / (self.opt.modalities_no + self.opt.seg_no)] *  (self.opt.modalities_no + self.opt.seg_no) #[0.2, 0.2, 0.2, 0.2, 0.2]

        self.loss_names = []
        self.visual_names = ['real_A']
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        for i in range(1, self.opt.modalities_no + self.opt.seg_no + 1): # + 1 because the range function does not inlcude the last number
            self.loss_names.extend(['G_GAN_' + str(i), 'G_L1_' + str(i), 'D_real_' + str(i), 'D_fake_' + str(i)])
            self.visual_names.extend(['fake_B_' + str(i), 'real_B_' + str(i)])

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.is_train:
            self.model_names = []
            for i in range(1, self.opt.modalities_no + 1):
                self.model_names.extend(['G' + str(i), 'D' + str(i)])
        else:  # during test time, only load G
            self.model_names = []
            for i in range(1, self.opt.modalities_no + 1):
                self.model_names.extend(['G' + str(i)])

            for i in range(1, self.opt.modalities_no + self.opt.input_no + 1):
                self.model_names.extend(['G5' + str(i)])

        # define networks (both generator and discriminator)
        for i in range(1, self.opt.modalities_no + 1):
            setattr(self, f'netG{i}', networks.define_G(opt.input_nc*opt.input_no, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.padding))

        if self.is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            for i in range(1, self.opt.modalities_no + 1):
                setattr(self, f'netD{i}', networks.define_D(opt.input_nc*opt.input_no + opt.output_nc , opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids))

        if self.is_train:
            # define loss functions
            self.criterionGAN_BCE = networks.GANLoss('vanilla').to(self.device)
            self.criterionGAN_lsgan = networks.GANLoss('lsgan').to(self.device)
            self.criterionSmoothL1 = torch.nn.SmoothL1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            for i in range(1, self.opt.modalities_no + 1):
                try:
                    params += list(getattr(self,f'netG{i}').parameters())
                except:
                    params = list(getattr(self,f'netG{i}').parameters())
            #params = list(self.netG1.parameters()) + list(self.netG2.parameters()) + list(self.netG3.parameters()) + list(self.netG4.parameters()) + list(self.netG51.parameters()) + list(self.netG52.parameters()) + list(self.netG53.parameters()) + list(self.netG54.parameters()) + list(self.netG55.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            del params
            
            for i in range(1, self.opt.modalities_no + 1):
                try:
                    params += list(getattr(self,f'netD{i}').parameters())
                except:
                    params = list(getattr(self,f'netD{i}').parameters())
            #params = list(self.netD1.parameters()) + list(self.netD2.parameters()) + list(self.netD3.parameters()) + list(self.netD4.parameters()) + list(self.netD51.parameters()) + list(self.netD52.parameters()) + list(self.netD53.parameters()) + list(self.netD54.parameters()) + list(self.netD55.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.criterionVGG = networks.VGGLoss().to(self.device)

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.

        :param input (dict): include the input image and the output modalities
        """

        self.real_A_array = input['A']
        As = [A.to(self.device) for A in self.real_A_array]
        self.real_A = torch.cat(As, dim=1) # shape: 1, (3 x input_no), 512, 512

        self.real_B_array = input['B']
        for i in range(self.opt.modalities_no):
            setattr(self, f'real_B_{i+1}', self.real_B_array[i].to(self.device))
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        for i in range(self.opt.modalities_no):
            setattr(self, f'fake_B_{i+1}', self.netG1(self.real_A))
        

    def backward_D(self):
        """Calculate GAN loss for the discriminators"""
        #dict_fake_AB = {f'fake_AB_{i}': torch.cat((self.real_A, getattr(self, f'fake_B_{i}')), 1) for i in range(1, self.opt.modalities_no + 1)}
        fake_AB_1 = torch.cat((self.real_A, self.fake_B_1), 1)  # Conditional GANs; feed IHC input and Hematoxtlin output to the discriminator
        fake_AB_2 = torch.cat((self.real_A, self.fake_B_2), 1)  # Conditional GANs; feed IHC input and mpIF DAPI output to the discriminator
        fake_AB_3 = torch.cat((self.real_A, self.fake_B_3), 1)  # Conditional GANs; feed IHC input and mpIF Lap2 output to the discriminator
        fake_AB_4 = torch.cat((self.real_A, self.fake_B_4), 1)  # Conditional GANs; feed IHC input and mpIF Ki67 output to the discriminator
        
        #dict_pred_fake = {f'pred_fake_{i}': getattr(self, f'netD{i}')(dict_fake_AB[f'fake_AB_{i}']) for i in range(1, self.opt.modalities_no + 1)}
        pred_fake_1 = self.netD1(fake_AB_1.detach())
        pred_fake_2 = self.netD2(fake_AB_2.detach())
        pred_fake_3 = self.netD3(fake_AB_3.detach())
        pred_fake_4 = self.netD4(fake_AB_4.detach())
        

        #for i in range(1, self.opt.modalities_no + 1):
        #    setattr(self, f'loss_D_fake_{i}', self.criterionGAN_BCE(dict_pred_fake[f'pred_fake_{i}'], False))
        #if self.opt.seg_gen:
        #    self.loss_D_fake_5 = self.criterionGAN_lsgan(pred_fake_5, False)
        self.loss_D_fake_1 = self.criterionGAN_BCE(pred_fake_1, False)
        self.loss_D_fake_2 = self.criterionGAN_BCE(pred_fake_2, False)
        self.loss_D_fake_3 = self.criterionGAN_BCE(pred_fake_3, False)
        self.loss_D_fake_4 = self.criterionGAN_BCE(pred_fake_4, False)

        #dict_real_AB = {f'real_AB_{i}':torch.cat((self.real_A, getattr(self, f'real_B_{i}')), 1) for i in range(1, self.opt.modalities_no + 1)}
        real_AB_1 = torch.cat((self.real_A, self.real_B_1), 1)
        real_AB_2 = torch.cat((self.real_A, self.real_B_2), 1)
        real_AB_3 = torch.cat((self.real_A, self.real_B_3), 1)
        real_AB_4 = torch.cat((self.real_A, self.real_B_4), 1)

        #dict_pred_real = {f'pred_real_{i}':getattr(self, f'netD{i}')(dict_real_AB[f'real_AB_{i}']) for i in range(1, self.opt.modalities_no + 1)}
        pred_real_1 = self.netD1(real_AB_1)
        pred_real_2 = self.netD2(real_AB_2)
        pred_real_3 = self.netD3(real_AB_3)
        pred_real_4 = self.netD4(real_AB_4)

        #for i in range(1, self.opt.modalities_no + 1):
        #    setattr(self, f'loss_D_real_{i}', self.criterionGAN_BCE(dict_pred_real[f'pred_real_{i}'], True))        
        #if self.opt.seg_gen:
        #    self.loss_D_real_5 = self.criterionGAN_lsgan(pred_real_5, True)
        self.loss_D_real_1 = self.criterionGAN_BCE(pred_real_1, True)
        self.loss_D_real_2 = self.criterionGAN_BCE(pred_real_2, True)
        self.loss_D_real_3 = self.criterionGAN_BCE(pred_real_3, True)
        self.loss_D_real_4 = self.criterionGAN_BCE(pred_real_4, True)

        # combine losses and calculate gradients
        #loss_D = 0
        #for i in range(1, self.opt.modalities_no + 1):
        #    loss_D += (getattr(self, f'loss_D_fake_{i}') + getattr(self, f'loss_D_real_{i}')) * 0.5 * self.loss_D_weights[i-1]
        #if self.opt.seg_gen:
        #    loss_D += (self.loss_D_fake_5 + self.loss_D_real_5) * 0.5 * self.loss_D_weights[-1]

        #self.loss_D = loss_D
        
        self.loss_D = (self.loss_D_fake_1 + self.loss_D_real_1) * 0.5 * self.loss_D_weights[0] + \
                      (self.loss_D_fake_2 + self.loss_D_real_2) * 0.5 * self.loss_D_weights[1] + \
                      (self.loss_D_fake_3 + self.loss_D_real_3) * 0.5 * self.loss_D_weights[2] + \
                      (self.loss_D_fake_4 + self.loss_D_real_4) * 0.5 * self.loss_D_weights[3]
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        
        #dict_fake_AB = {f'fake_AB_{i}': torch.cat((self.real_A, getattr(self, f'fake_B_{i}')), 1) for i in range(1, self.opt.modalities_no + 1)}
        fake_AB_1 = torch.cat((self.real_A, self.fake_B_1), 1)
        fake_AB_2 = torch.cat((self.real_A, self.fake_B_2), 1)
        fake_AB_3 = torch.cat((self.real_A, self.fake_B_3), 1)
        fake_AB_4 = torch.cat((self.real_A, self.fake_B_4), 1)
        
        #dict_pred_fake = {f'pred_fake_{i}': getattr(self, f'netD{i}')(dict_fake_AB[f'fake_AB_{i}']) for i in range(1, self.opt.modalities_no + 1)}
        pred_fake_1 = self.netD1(fake_AB_1)
        pred_fake_2 = self.netD2(fake_AB_2)
        pred_fake_3 = self.netD3(fake_AB_3)
        pred_fake_4 = self.netD4(fake_AB_4)
        
        
        #for i in range(1, self.opt.modalities_no + 1):
        #    setattr(self, f'loss_G_GAN_{i}', self.criterionGAN_BCE(dict_pred_fake[f'pred_fake_{i}'], True))
        #if self.opt.seg_gen:
        #    self.loss_G_GAN_5 = self.criterionGAN_lsgan(pred_fake_5, True)
        self.loss_G_GAN_1 = self.criterionGAN_BCE(pred_fake_1, True)
        self.loss_G_GAN_2 = self.criterionGAN_BCE(pred_fake_2, True)
        self.loss_G_GAN_3 = self.criterionGAN_BCE(pred_fake_3, True)
        self.loss_G_GAN_4 = self.criterionGAN_BCE(pred_fake_4, True)

        # Second, G(A) = B
        #for i in range(1, self.opt.modalities_no + 1):
        #    setattr(self, f'loss_G_L1_{i}', self.criterionSmoothL1(getattr(self, f'fake_B_{i}'), getattr(self, f'real_B_{i}')) * self.opt.lambda_L1)
        #if self.opt.seg_gen:
        #    self.loss_G_L1_5 = self.criterionSmoothL1(self.fake_B_5, self.real_B_5) * self.opt.lambda_L1
        self.loss_G_L1_1 = self.criterionSmoothL1(self.fake_B_1, self.real_B_1) * self.opt.lambda_L1
        self.loss_G_L1_2 = self.criterionSmoothL1(self.fake_B_2, self.real_B_2) * self.opt.lambda_L1
        self.loss_G_L1_3 = self.criterionSmoothL1(self.fake_B_3, self.real_B_3) * self.opt.lambda_L1
        self.loss_G_L1_4 = self.criterionSmoothL1(self.fake_B_4, self.real_B_4) * self.opt.lambda_L1

        #for i in range(1, self.opt.modalities_no + 1):
        #    setattr(self, f'loss_G_VGG_{i}', self.criterionVGG(getattr(self, f'fake_B_{i}'), getattr(self, f'real_B_{i}')) * self.opt.lambda_feat)
        self.loss_G_VGG_1 = self.criterionVGG(self.fake_B_1, self.real_B_1) * self.opt.lambda_feat
        self.loss_G_VGG_2 = self.criterionVGG(self.fake_B_2, self.real_B_2) * self.opt.lambda_feat
        self.loss_G_VGG_3 = self.criterionVGG(self.fake_B_3, self.real_B_3) * self.opt.lambda_feat
        self.loss_G_VGG_4 = self.criterionVGG(self.fake_B_4, self.real_B_4) * self.opt.lambda_feat
        
        #loss_G = 0
        #for i in range(1, self.opt.modalities_no + 1):
        #    loss_G += (getattr(self, f'loss_G_GAN_{i}') + getattr(self, f'loss_G_L1_{i}') + getattr(self, f'loss_G_VGG_{i}')) * self.loss_G_weights[i-1]
        #if self.opt.seg_gen:
        #    loss_G += (self.loss_G_GAN_5 + self.loss_G_L1_5) * self.loss_G_weights[4]
        #print(loss_G)
        #self.loss_G = loss_G
        #del loss_G
        
        self.loss_G = (self.loss_G_GAN_1 + self.loss_G_L1_1 + self.loss_G_VGG_1) * self.loss_G_weights[0] + \
                      (self.loss_G_GAN_2 + self.loss_G_L1_2 + self.loss_G_VGG_2) * self.loss_G_weights[1] + \
                      (self.loss_G_GAN_3 + self.loss_G_L1_3 + self.loss_G_VGG_3) * self.loss_G_weights[2] + \
                      (self.loss_G_GAN_4 + self.loss_G_L1_4 + self.loss_G_VGG_4) * self.loss_G_weights[3]
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        #for i in range(1, self.opt.modalities_no + 1):
        #    self.set_requires_grad(getattr(self, f'netD{i}'), True)
        #if self.opt.seg_gen:
        #    for i in range(1, self.opt.modalities_no + self.opt.seg_no + 1):
        #        self.set_requires_grad(getattr(self, f'netD5{i}'), True)
        self.set_requires_grad(self.netD1, True)  # enable backprop for D1
        self.set_requires_grad(self.netD2, True)  # enable backprop for D2
        self.set_requires_grad(self.netD3, True)  # enable backprop for D3
        self.set_requires_grad(self.netD4, True)  # enable backprop for D4

        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        # update G
        #for i in range(1, self.opt.modalities_no + 1):
        #    self.set_requires_grad(getattr(self, f'netD{i}'), False)
        #if self.opt.seg_gen:
        #    for i in range(1, self.opt.modalities_no + self.opt.seg_no + 1):
        #        self.set_requires_grad(getattr(self, f'netD5{i}'), False)
        self.set_requires_grad(self.netD1, False)  # D1 requires no gradients when optimizing G1
        self.set_requires_grad(self.netD2, False)  # D2 requires no gradients when optimizing G2
        self.set_requires_grad(self.netD3, False)  # D3 requires no gradients when optimizing G3
        self.set_requires_grad(self.netD4, False)  # D4 requires no gradients when optimizing G4

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
