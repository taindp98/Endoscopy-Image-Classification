'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

# taken from https://github.com/kuangliu/pytorch-cifar
import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.models.models.cbam import CBAM

import torch.nn as nn   
import torch
import torch.nn.functional as F



class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio, kernel_size):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att = self.spatial_attention(fp)
        # print(spat_att.size())
        fpp = spat_att * fp
        # print(fpp.size())
        return fpp


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim = 1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1,x.size()[1],1,1)
        att = torch.sigmoid(conv)        
        return att

    def agg_channel(self, x, pool = "max"):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == "max":
            x = F.max_pool1d(x,c)
        elif pool == "avg":
            x = F.avg_pool1d(x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)

        
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1,1,kernel[0], kernel[1])
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, reduction_ratio = 1, kernel_cbam = 3, use_cbam = False):
        super(BasicBlock, self).__init__()
        self.use_cbam = use_cbam
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if self.use_cbam:
            self.cbam = CBAM(n_channels_in = self.expansion*planes, reduction_ratio = reduction_ratio, kernel_size = kernel_cbam)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        #cbam
        if self.use_cbam:
            out = self.cbam(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, reduction_ratio = 1, kernel_cbam = 3, use_cbam = False):
        super(Bottleneck, self).__init__()
        self.use_cbam = use_cbam

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        if self.use_cbam:
            self.cbam = CBAM(n_channels_in = self.expansion*planes, reduction_ratio = reduction_ratio, kernel_size = kernel_cbam)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        #cbam
        if self.use_cbam:
            out = self.cbam(out)

        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNetCBAM(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, reduction_ratio = 1, kernel_cbam = 3, use_cbam_block= False, use_cbam_class = False):
        super(ResNetCBAM, self).__init__()
        self.in_planes = 64
        self.reduction_ratio = reduction_ratio
        self.kernel_cbam = kernel_cbam
        self.use_cbam_block = use_cbam_block
        self.use_cbam_class = use_cbam_class

        print(use_cbam_block, use_cbam_class)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        if self.use_cbam_class:
            self.cbam = CBAM(n_channels_in = 512*block.expansion, reduction_ratio = reduction_ratio, kernel_size = kernel_cbam)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.reduction_ratio, self.kernel_cbam, self.use_cbam_block))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.use_cbam_class:
            out = out  + self.cbam(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return out




# def ResNet18(reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False):
#     print(kernel_cbam)
#     return ResNet(
#                 BasicBlock, 
#                 [2,2,2,2], 
#                 reduction_ratio= reduction_ratio,
#                 kernel_cbam = kernel_cbam,
#                 use_cbam_block= use_cbam_block,
#                 use_cbam_class = use_cbam_class
#                 )

# def ResNet34(reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False):
#     return ResNet(
#         BasicBlock,
#         [3,4,6,3],
#         reduction_ratio= reduction_ratio, 
#         kernel_cbam = kernel_cbam, 
#         use_cbam_block= use_cbam_block,
#         use_cbam_class = use_cbam_class
#         )

# def ResNet50(reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False):
#     return ResNet(
#         Bottleneck, 
#         [3,4,6,3], 
#         reduction_ratio= reduction_ratio, 
#         kernel_cbam = kernel_cbam, 
#         use_cbam_block= use_cbam_block,
#         use_cbam_class = use_cbam_class
#         )

# def ResNet101(reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False):
#     return ResNet(
#         Bottleneck, 
#         [3,4,23,3], 
#         reduction_ratio= reduction_ratio, 
#         kernel_cbam = kernel_cbam, 
#         use_cbam_block= use_cbam_block,
#         use_cbam_class = use_cbam_class
#         )

# def ResNet152(reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False):
#     return ResNet(
#         Bottleneck, 
#         [3,8,36,3], 
#         reduction_ratio= reduction_ratio, 
#         kernel_cbam = kernel_cbam, 
#         use_cbam_block= use_cbam_block,
#         use_cbam_class = use_cbam_class
#         )

# def ResNetk(k, reduction_ratio = 1, kernel_cbam = 3, use_cbam_block = False, use_cbam_class = False ):
#     possible_depth = [18,34,50,101,152]
#     assert k in possible_depth, "Choose a depth in {}".format(possible_depth)

#     if k == 18:
#         return ResNet18(reduction_ratio= reduction_ratio, kernel_cbam = kernel_cbam, use_cbam_block= use_cbam_block, use_cbam_class = use_cbam_class)
#     elif k == 34:
#         return ResNet34(reduction_ratio= reduction_ratio, kernel_cbam = kernel_cbam, use_cbam_block= use_cbam_block, use_cbam_class = use_cbam_class)
#     elif k == 50:
#         return ResNet50(reduction_ratio= reduction_ratio, kernel_cbam = kernel_cbam, use_cbam_block= use_cbam_block, use_cbam_class = use_cbam_class)
#     elif k == 101:
#         return ResNet101(reduction_ratio= reduction_ratio, kernel_cbam = kernel_cbam, use_cbam_block= use_cbam_block, use_cbam_class = use_cbam_class)
#     elif k == 152:
#         return ResNet152(reduction_ratio= reduction_ratio, kernel_cbam = kernel_cbam, use_cbam_block= use_cbam_block, use_cbam_class = use_cbam_class)


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1,3,32,32))
#     print(y.size())

# test()