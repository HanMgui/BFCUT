'''Codes come from ContrastiveCrop-main,
    https://arxiv.org/abs/2202.03278
'''
import torch
import torch.nn as nn
import random
from torch.nn import init

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, zero_init_residual=False, maxpool=True):
        '''mask_model: 对q掩膜的方式,如[x,maskx1,maskx2],等'''
        super().__init__()
        depth = 50
        blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3],
                  200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

        self.maxpl = maxpool
        self.inplanes = 64
        if maxpool:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))    #hmg 去掉
        # self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, layers=[]):
        if len(layers) > 0:
            feats = []
            if 0 in layers: feats.append(x)
            # bs = x.shape[0]#256
            x = self.conv1(x)#128  1,64,128,128
            x = self.bn1(x)
            x = self.relu(x)
            if self.maxpl:
                x = self.maxpool(x)#经过这行后变64  1,64,64,64

            x = self.layer1(x)   # 1,256,64,64
            if 1 in layers: feats.append(x)
            x = self.layer2(x)#经过这行后变32  # 1,512,32,32
            if 2 in layers: feats.append(x)
            x = self.layer3(x)   # 1,1024,16,16
            if 3 in layers: feats.append(x)
            x = self.layer4(x)  # 1,2048,8,8
            if 4 in layers: feats.append(x)
            return x, feats
        else:
            # bs = x.shape[0]#256
            x = self.conv1(x)#128
            x = self.bn1(x)
            x = self.relu(x)
            if self.maxpl:
                x = self.maxpool(x)#经过这行后变64

            x = self.layer1(x)
            x = self.layer2(x)#经过这行后变32
            x = self.layer3(x)
            x = self.layer4(x)

            # x = self.avgpool(x)  #hmg 去掉
            # x = torch.flatten(x, 1)#hmg 去掉
            # x = self.fc(x)#hmg 去掉
            return x

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y
    
class SEResnet(nn.Module):
    def __init__(self,opt,in_channels=2048,othermodel=None,device='cpu'):
        super(SEResnet,self).__init__()
        self.device = device
        self.SE = SEModule(in_channels).to(self.device)
        self._init_net(self.SE,opt.init_type,opt.init_gain,opt.gpu_ids)
        self.model = othermodel
        if self.model == None:
            self.model = ResNet().to(self.device)
            self.loadres()
    
    def loadres(self): #加载resnet50的参数
        corresresnet={1:['moco_v2_800ep_pretrain.pth.tar','state_dict','module.encoder_q.','***'],
                    #   2:['pixpro_base_r50_400ep_md5_919c6612.pth','model','module.encoder','_k']
                      }
        whichresnet=1
        resnet50_path = 'checkpoints/ResNet50/'+corresresnet[whichresnet][0]
        print('load %s parameters'%(resnet50_path))

        state_dict1 = torch.load(resnet50_path, map_location=str(self.device))#ResNet50
        state_dict1=state_dict1[corresresnet[whichresnet][1]]
        if hasattr(state_dict1, '_metadata'):
            del state_dict1._metadata
        import collections
        state_dict=collections.OrderedDict()
        keys1 = list(state_dict1.keys())  #resnet的
        for k in range(len(keys1)):  #删掉没用的，比如_k
            if keys1[k].find(corresresnet[whichresnet][3])>=0:
                del state_dict1[keys1[k]]
        keys1 = list(state_dict1.keys())  #resnet的
        for k in range(len(keys1)):
            if keys1[k].find(corresresnet[whichresnet][2])>=0:  #只要corresresnet[whichresnet][2]
                state_dict[keys1[k].replace(corresresnet[whichresnet][2],'')]=state_dict1[keys1[k]]

        net = self.model
        net_keys = set(net.state_dict().keys())
        state_keys = set(state_dict.keys())
        print('resnet load parameters has some error:')
        print('--net - state:%s'%(net_keys-state_keys))
        print('--state - net:%s'%(state_keys-net_keys))
        net.load_state_dict(state_dict,strict=False) #hmg改，没有加载adjust层的参数

        for param in net.parameters():#冻结resnet的参数
            param.requires_grad = False

    def forward(self,x,se=True):
        x = self.model(x)
        if se:x = self.SE(x)   #用于 训练时保留，画热图时选择性去掉
        # x = torch.nn.Sigmoid(x.sum(dim=1,keepdim=True))
        x = x.sum(dim=1,keepdim=True)
        return x  #n,1,8,8

    
    def _init_net(self,net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
        if initialize_weights:
            self._init_weights(net, init_type, init_gain=init_gain, debug=debug)
        return net
        
    def _init_weights(self, net, init_type='normal', init_gain=0.02, debug=False):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if debug:
                    print(classname)
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        net.apply(init_func)  # apply the initialization function <init_func>

