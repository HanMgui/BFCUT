import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import os

from torchvision.utils import save_image#保存张量为图片

import matplotlib.pyplot as plt
import shutil

import torch.nn as nn


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        if self.opt.lambda_NCE == 0: #hmg改，为了不让这个损失计算
            self.loss_names.remove('NCE')
            self.loss_names.remove('NCE_Y')

        if self.opt.usehmgmodification:
            if self.opt.perceptual_alph>0:
                self.loss_names += ['per']
            # if self.opt.cam_alph>0:
            #     self.loss_names += ['cam']

        # define networks (both generator and discriminator)
        if self.opt.usehmgmodification and self.opt.useresnet:#hmg改，改变G的网络结构
            # if self.isTrain: self.model_names.remove('F')
            self.netG = networks.define_G_encdec('resnet50', opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt) 
            self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        else:
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
            self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.opt.usehmgmodification and self.opt.cam_alph>0:#hmg加，加上热图
            from . import resnet_gai
            if opt.cam_name=='resnet':othermodel=None
            self.netSE = resnet_gai.SEResnet(self.opt,othermodel=othermodel,device=self.device)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if self.opt.usehmgmodification:
                if self.opt.cam_alph>0:
                    self.model_names.append('SE')
                    self.optimizer_SE = torch.optim.Adam(self.netSE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                    self.queue_k = torch.rand(size=[256,256],device=self.device)
                    self.queue_len = 256-256%self.opt.batch_size
                    self.queue_ptr = 0

                if self.opt.perceptual_alph>0:
                    from torchvision.models import vgg16
                    from .perceptual import LossNetwork
                    vgg_model = vgg16(pretrained=True).features[:16]
                    vgg_model = vgg_model.to(self.device)
                    # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
                    for param in vgg_model.parameters():
                        param.requires_grad = False
                    self.perceptual_net=LossNetwork(vgg_model)
                    self.perceptual_net.eval()

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        # if self.opt.netF == 'mlp_sample':
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:#hmg改  
            self.optimizer_F.zero_grad()
        if self.opt.usehmgmodification and self.opt.cam_alph>0:
            self.optimizer_SE.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        # if self.opt.netF == 'mlp_sample':
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:#hmg改   
            self.optimizer_F.step()
        if self.opt.usehmgmodification and self.opt.cam_alph>0:
            self.optimizer_SE.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_cam=0
            # if self.opt.usehmgmodification and self.opt.cam_alph>0:
            #     self.loss_NCE, self.loss_cam = self.calculate_NCE_loss(self.real_A, self.fake_B,saveim=False)
            # else:
                # self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B,saveim=False)
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B,saveim=False)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss_idt(self.real_B, self.idt_B) 
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE
        self.loss_G = 0
        if self.opt.usehmgmodification:
            if self.opt.perceptual_alph>0: #hmg添加
                self.loss_per= self.perceptual_net(self.fake_B, self.real_A)*self.opt.perceptual_alph
                self.loss_G+=self.loss_per

        self.loss_G += self.loss_G_GAN + loss_NCE_both + self.loss_cam
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt,saveim=False):
        n_layers = len(self.nce_layers)
        if self.opt.usehmgmodification and self.opt.cam_alph>0: # 取得源图的cam和第20层的q的特征图并相乘
            feat_q = self.netG(tgt, self.nce_layers+[20], encode_only=True)
            qshape=feat_q[-1].shape
            # cam = self.netSE(src)#/cam.max(dim=2,keepdim=True)[0].max(dim=3,keepdim=True)[0] 省略了0  (x-0)/(max-0)
            cam = self.netSE(src) #max-cam+0.1
            cam = cam.max(dim=2,keepdim=True)[0].max(dim=3,keepdim=True)[0] - cam + 0.1
            if self.opt.useminmaxq:
                cam_sortedid=cam.view(self.opt.batch_size,1,-1).sort()[1]#index  [batch,1,8*8],从小到大的顺序
                minmaxid=torch.cat([cam_sortedid[:,0,:8],cam_sortedid[:,0,-8:]],dim=1)
                # minmaxid=cam_sortedid[:,0,:16]
                minmaxidxy=torch.ones(size=[minmaxid.shape[0],2,minmaxid.shape[1]],device=self.device)
                minmaxidxy[:,0,:] = minmaxid // cam.shape[3] + 0.5 #x  +0.5是为了取放大图中靠近中间的点
                minmaxidxy[:,1,:] = minmaxid % cam.shape[2] + 0.5  #y

            cam = torch.nn.functional.interpolate(cam,size=[qshape[2],qshape[3]]) 
            feat_q[-1] = torch.mul(feat_q[-1],cam).sum(dim=[2,3],keepdim=True)   #.sum(dim=[2,3],keepdim=True)如果是使用像素级损失的话应该是没有这一项才对
            # feat_q[-1] = feat_q[-1].sum(dim=[2,3],keepdim=True)   #.sum(dim=[2,3],keepdim=True)如果是使用像素级损失的话应该是没有这一项才对
        else:
            feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        if saveim:#hmg加，用于保存中间特征图以查看resnet能不能定位前景
            for i in range(len(feat_q)):
                if i==0:shutil.copy(self.image_paths[0],'./results/%s'%(self.image_paths[0].split('/')[-1]))
                else:
                    im = feat_q[i][0].sum(dim=0).detach().to('cpu').numpy()
                    plt.imsave('./results/%s_%d.jpg'%(self.image_paths[0].split('/')[-1].split('.')[0],i),im)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        if self.opt.usehmgmodification and self.opt.cam_alph>0: # 取得第20层的k的特征图并相乘
            feat_k = self.netG(src, self.nce_layers+[20], encode_only=True)
            feat_k[-1] = torch.mul(feat_k[-1],cam).sum(dim=[2,3],keepdim=True)
            # feat_k[-1] = feat_k[-1].sum(dim=[2,3],keepdim=True)
        else:
            feat_k = self.netG(src, self.nce_layers, encode_only=True)
        if self.opt.usehmgmodification and self.opt.cam_alph>0 and self.opt.useminmaxq:
            feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None,minmaxidxy=minmaxidxy)
        else:
            feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()
        if self.opt.usehmgmodification and self.opt.cam_alph>0: # 取得源图的cam和第20层的q的特征图并相乘
            self.pullandpush_queue(feat_k_pool[-1])
            loss = self.contrastive_loss(feat_q_pool[-1], self.queue_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    def calculate_NCE_loss_idt(self, src, tgt):  # hmg加，为了减少一点计算量(不用在不需要计算cam的地方再判断一次是不是要用cam)
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    def setup(self, opt): #hmg添加，加载不同的预训练参数
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            if self.opt.cam_alph>0:
                self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
                self.schedulers.append(networks.get_scheduler(self.optimizer_SE, opt,cam=opt.lr_policy)) #cam=opt.lr_policy,'',''
            else:
                self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        load_suffix = opt.epoch
        if opt.usehmgmodification and opt.useresnet:
            continue_train = True
            if self.isTrain and not opt.continue_train:#除了从头训练需要结合不同组件的参数外，其余情况只需要加载保存好的参数
                continue_train = False 
            self.load_multi_networks(load_suffix,continue_train)
            if self.isTrain: self.set_requires_grad_layers(False) #hmg添加用于冻结某些层的参数
        else: # CUT的加载参数
            if not self.isTrain or opt.continue_train:
                self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def load_multi_networks(self, epoch, continue_train=True): #hmg添加，用于加载多个模型的参数
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if not continue_train:
            def _replace_number(keyname):
                # 21--0 22--1
                yuan_num = 21
                huan_num = 0
                for i in range(10):
                    if keyname.find('.'+str(yuan_num+i))>0:
                        keyname = keyname.replace('.'+str(yuan_num+i),'.'+str(huan_num+i))
                        return keyname
                return keyname
            corresresnet={1:['moco_v2_800ep_pretrain.pth.tar','state_dict','module.encoder_q','***'],
                          2:['pixpro_base_r50_400ep_md5_919c6612.pth','model','module.encoder','_k']}
            whichresnet=1
            resnet50_path = 'checkpoints/ResNet50/'+corresresnet[whichresnet][0]
            netG_path = 'checkpoints/H2Z_CUT_yuan/400_net_G.pth'
            netD_path = 'checkpoints/H2Z_CUT_yuan/400_net_D.pth'
            print('load %s       parameters\n     %s\n     %s'%(resnet50_path,netG_path,netD_path))
            for name in self.model_names:
                if isinstance(name, str):  
                    state_dict=None                  
                    if name=='G':
                        state_dict1 = torch.load(resnet50_path, map_location=str(self.device))#ResNet50
                        state_dict1=state_dict1[corresresnet[whichresnet][1]]
                        if hasattr(state_dict1, '_metadata'):
                            del state_dict1._metadata
                        import collections
                        state_dict=collections.OrderedDict()
                        # state_dict2 = torch.load(netG_path, map_location=str(self.device))#CUT
                        # if hasattr(state_dict2, '_metadata'):
                        #     del state_dict2._metadata                        
                        # load_name1=['model.1.weight','model.20.conv_block.5.bias']  #包含左右的
                        load_name2=['model.21.filt','model.30.bias']
                        keys1 = list(state_dict1.keys())  #resnet的
                        # keys2 = list(state_dict2.keys())  #GAN的
                        for k in range(len(keys1)):  #删掉没用的，比如_k
                            if keys1[k].find(corresresnet[whichresnet][3])>=0:
                                del state_dict1[keys1[k]]
                        keys1 = list(state_dict1.keys())  #resnet的
                        for k in range(len(keys1)):
                            if keys1[k].find(corresresnet[whichresnet][2])>=0:  #只要corresresnet[whichresnet][2]
                                state_dict[keys1[k].replace(corresresnet[whichresnet][2],'enc')]=state_dict1[keys1[k]]
                        # kstart=keys2.index(load_name2[0])
                        # for k in range(kstart,len(keys2)):
                            # state_dict['dec.'+_replace_number(keys2[k])]=state_dict2[keys2[k]]
                    # elif name=='D':
                    #     state_dict = torch.load(netD_path, map_location=str(self.device))
                    #     if hasattr(state_dict, '_metadata'):
                    #         del state_dict._metadata
                    if  state_dict is not None:
                        net = getattr(self, 'net' + name)
                        net_keys = set(net.state_dict().keys())
                        state_keys = set(state_dict.keys())
                        print('net'+name+' load parameters has some error:')
                        print('--net - state:%s'%(net_keys-state_keys))
                        print('--state - net:%s'%(state_keys-net_keys))
                        net.load_state_dict(state_dict,strict=False) #hmg改，没有加载adjust层的参数
        else: #根据保存的pth加载
            for name in self.model_names:
                if isinstance(name, str):
                    load_filename = '%s_net_%s.pth' % (epoch, name)
                    if self.opt.isTrain and self.opt.pretrained_name is not None:
                        load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
                    else:
                        load_dir = self.save_dir
                    load_path = os.path.join(load_dir, load_filename)
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                    net.load_state_dict(state_dict)
                
    def set_requires_grad_layers(self, requires_grad=False):#hmg添加，用于冻结某些部件的参数
        names = [      'G.enc'] #为了输出方便      'D',,      'G.dec'
        nets = [self.netG.enc] #self.netD,,self.netG.enc,
        print('frozen %s parameters'%names)
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def pullandpush_queue(self,feat_k):
        self.queue_k[self.queue_ptr:self.queue_ptr+self.opt.batch_size] = feat_k.detach()
        self.queue_ptr += self.opt.batch_size
        if self.queue_ptr > self.queue_len - self.opt.batch_size: 
            self.queue_ptr = 0 
    
    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.opt.nce_T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.opt.nce_T)