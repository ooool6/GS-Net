import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from collections import OrderedDict

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class SeparableConv2d_aspp(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, padding=0,
                 normalization='bn', num_groups=8):
        super(SeparableConv2d_aspp, self).__init__()
        self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                                   groups=inplanes, bias=bias)
        if normalization == 'bn':
            self.depthwise_bn = nn.BatchNorm2d(inplanes)
        elif normalization == 'gn':
            self.depthwise_bn = nn.GroupNorm(num_groups=num_groups, num_channels=inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        if normalization == 'bn':
            self.pointwise_bn = nn.BatchNorm2d(planes)
        elif normalization == 'gn':
            self.pointwise_bn = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)
        return x

class Decoder_module(nn.Module):
    def __init__(self, inplanes, planes, rate=1, normalization='bn', num_groups=8):
        super(Decoder_module, self).__init__()
        self.atrous_convolution = SeparableConv2d_aspp(inplanes, planes, 3, stride=1, dilation=rate, padding=1,
                                                       normalization=normalization, num_groups=num_groups)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate, normalization='bn', num_groups=8):
        super(ASPP_module, self).__init__()
        if rate == 1:
            raise RuntimeError()
        else:
            kernel_size = 3
            padding = rate
            self.atrous_convolution = SeparableConv2d_aspp(inplanes, planes, kernel_size, stride=1, dilation=rate,
                                                           padding=padding, normalization=normalization, num_groups=num_groups)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x

class ASPP_module_rate0(nn.Module):
    def __init__(self, inplanes, planes, rate=1, normalization='bn', num_groups=8):
        super(ASPP_module_rate0, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
            self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                                stride=1, padding=padding, dilation=rate, bias=False)
            if normalization == 'bn':
                self.bn = nn.BatchNorm2d(planes, eps=1e-5, affine=True)
            elif normalization == 'gn':
                self.bn = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.relu = nn.ReLU()
        else:
            raise RuntimeError()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, padding=0,
                 normalization='bn', num_groups=8):
        super(SeparableConv2d_same, self).__init__()
        if planes % num_groups != 0:
            num_groups = int(num_groups / 2)
        if inplanes % num_groups != 0:
            num_groups = int(num_groups / 2)
        self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                                   groups=inplanes, bias=bias)
        if normalization == 'bn':
            self.depthwise_bn = nn.BatchNorm2d(inplanes)
        elif normalization == 'gn':
            self.depthwise_bn = nn.GroupNorm(num_groups=num_groups, num_channels=inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        if normalization == 'bn':
            self.pointwise_bn = nn.BatchNorm2d(planes)
        elif normalization == 'gn':
            self.pointwise_bn = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

    def forward(self, x):
        x = fixed_padding(x, self.depthwise.kernel_size[0], rate=self.depthwise.dilation[0])
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        return x

class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True,
                 is_last=False, normalization='bn', num_groups=8):
        super(Block, self).__init__()
        if planes % num_groups != 0:
            num_groups = int(num_groups / 2)
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            if is_last:
                self.skip = nn.Conv2d(inplanes, planes, 1, stride=1, bias=False)
            if normalization == 'bn':
                self.skipbn = nn.BatchNorm2d(planes)
            elif normalization == 'gn':
                self.skipbn = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=stride, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))

        if is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x

class Block2(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True,
                 is_last=False, normalization='bn', num_groups=8):
        super(Block2, self).__init__()
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            if normalization == 'bn':
                self.skipbn = nn.BatchNorm2d(planes)
            elif normalization == 'gn':
                self.skipbn = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation,
                                            normalization=normalization, num_groups=num_groups))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            self.block2_lastconv = nn.Sequential(*[self.relu, SeparableConv2d_same(planes, planes, 3, stride=stride,
                                                                                  dilation=dilation,
                                                                                  normalization=normalization,
                                                                                  num_groups=num_groups)])

        if is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1, normalization=normalization, num_groups=num_groups))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        low_middle = x.clone()
        x1 = x
        x1 = self.block2_lastconv(x1)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x1 += skip
        return x1, low_middle

class Xception(nn.Module):
    def __init__(self, inplanes=1, os=16, pretrained=False, normalization='bn', num_groups=8):
        super(Xception, self).__init__()
        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(inplanes, 16, 3, stride=2, padding=1, bias=False)
        if normalization == 'bn':
            self.bn1 = nn.BatchNorm2d(16)
        elif normalization == 'gn':
            self.bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=16)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False)
        if normalization == 'bn':
            self.bn2 = nn.BatchNorm2d(32)
        elif normalization == 'gn':
            self.bn2 = nn.GroupNorm(num_groups=num_groups, num_channels=32)

        self.block1 = Block(32, 64, reps=2, stride=2, start_with_relu=False, grow_first=True,
                            normalization=normalization, num_groups=num_groups)
        self.block2 = Block2(64, 128, reps=2, stride=2, start_with_relu=True, grow_first=True,
                             normalization=normalization, num_groups=num_groups)
        self.block3 = Block(128, 364, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True,
                            normalization=normalization, num_groups=num_groups)

        self.block20 = Block(364, 512, reps=2, stride=1, dilation=exit_block_rates[0],
                             start_with_relu=True, grow_first=False, is_last=True,
                             normalization=normalization, num_groups=num_groups)

        self.conv3 = SeparableConv2d_aspp(512, 768, 3, stride=1, dilation=exit_block_rates[1], padding=exit_block_rates[1],
                                          normalization=normalization, num_groups=num_groups)
        self.conv4 = SeparableConv2d_aspp(768, 768, 3, stride=1, dilation=exit_block_rates[1], padding=exit_block_rates[1],
                                          normalization=normalization, num_groups=num_groups)
        self.conv5 = SeparableConv2d_aspp(768, 1024, 3, stride=1, dilation=exit_block_rates[1], padding=exit_block_rates[1],
                                          normalization=normalization, num_groups=num_groups)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        low_level_feat_2 = self.relu(x)
        x = self.block1(low_level_feat_2)
        x, low_level_feat_4 = self.block2(x)
        x = self.block3(x)

        x = self.block20(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        return x, low_level_feat_2, low_level_feat_4

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __load_xception_pretrained(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class DeepLabv3_plus_skipconnection_2d(nn.Module):
    def __init__(self, nInputChannels=1, n_classes=17, os=16, pretrained=False, _print=True, normalization='bn', num_groups=8):
        if _print:
            print(f"Constructing DeepLabv3+ model...")
            print(f"Number of classes: {n_classes}")
            print(f"Output stride: {os}")
            print(f"Number of Input Channels: {nInputChannels}")
        super(DeepLabv3_plus_skipconnection_2d, self).__init__()
        self.in_channels = nInputChannels
        self.use_unary = (nInputChannels > 1)
        self.xception_features = Xception(nInputChannels, os, pretrained, normalization=normalization, num_groups=num_groups)

        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module_rate0(1024, 128, rate=rates[0], normalization=normalization, num_groups=num_groups)
        self.aspp2 = ASPP_module(1024, 128, rate=rates[1], normalization=normalization, num_groups=num_groups)
        self.aspp3 = ASPP_module(1024, 128, rate=rates[2], normalization=normalization, num_groups=num_groups)
        self.aspp4 = ASPP_module(1024, 128, rate=rates[3], normalization=normalization, num_groups=num_groups)

        self.relu = nn.ReLU()
        if normalization == 'bn':
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(1024, 128, 1, stride=1, bias=False),
                                                 nn.BatchNorm2d(128),
                                                 nn.ReLU())
        elif normalization == 'gn':
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(1024, 128, 1, stride=1, bias=False),
                                                 nn.GroupNorm(num_groups=num_groups, num_channels=128),
                                                 nn.ReLU())

        self.concat_projection_conv1 = nn.Conv2d(640, 128, 1, bias=False)
        if normalization == 'bn':
            self.concat_projection_bn1 = nn.BatchNorm2d(128)
        elif normalization == 'gn':
            self.concat_projection_bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=128)

        self.feature_projection_conv1 = nn.Conv2d(128, 24, 1, bias=False)
        if normalization == 'bn':
            self.feature_projection_bn1 = nn.BatchNorm2d(24)
        elif normalization == 'gn':
            self.feature_projection_bn1 = nn.GroupNorm(num_groups=num_groups, num_channels=24)

        self.feature_projection_conv2 = nn.Conv2d(32, 64, 1, bias=False)
        if normalization == 'bn':
            self.feature_projection_bn2 = nn.BatchNorm2d(64)
        elif normalization == 'gn':
            self.feature_projection_bn2 = nn.GroupNorm(num_groups=num_groups, num_channels=64)

        self.decoder1 = nn.Sequential(Decoder_module(152, 128, normalization=normalization, num_groups=num_groups),
                                      Decoder_module(128, 128, normalization=normalization, num_groups=num_groups))
        self.decoder2 = nn.Sequential(Decoder_module(192, 256, normalization=normalization, num_groups=num_groups),
                                      Decoder_module(256, 256, normalization=normalization, num_groups=num_groups))
        self.semantic = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)

        self.final_activation = nn.Sigmoid() if n_classes == 2 else nn.Softmax(dim=1)

    def forward(self, img, unary=None, **kwargs):
        input = img
        if self.use_unary and unary is not None:
            pre_seg = F.interpolate(unary, size=input.shape[2:], mode='bilinear', align_corners=True)
            input = torch.cat((input, pre_seg), dim=1)
        x, low_level_features_2, low_level_features_4 = self.xception_features(input)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)

        low_level_features_2 = self.feature_projection_conv2(low_level_features_2)
        low_level_features_2 = self.feature_projection_bn2(low_level_features_2)
        low_level_features_2 = self.relu(low_level_features_2)

        low_level_features_4 = self.feature_projection_conv1(low_level_features_4)
        low_level_features_4 = self.feature_projection_bn1(low_level_features_4)
        low_level_features_4 = self.relu(low_level_features_4)

        x = F.interpolate(x, size=low_level_features_4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features_4), dim=1)
        x = self.decoder1(x)
        x = F.interpolate(x, size=low_level_features_2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features_2), dim=1)
        x = self.decoder2(x)
        x = self.semantic(x)
        final_conv = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        if not self.training:
            x = self.final_activation(final_conv)
            return x, final_conv, None
        else:
            return final_conv, None

    def freeze_bn(self):
        for m in self.xception_features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_totally_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_aspp_bn(self):
        for m in self.aspp1.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.aspp2.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.aspp3.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        for m in self.aspp4.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def learnable_parameters(self):
        layer_features_BN = []
        layer_features = []
        layer_aspp = []
        layer_projection = []
        layer_decoder = []
        layer_other = []
        model_para = list(self.named_parameters())
        for name, para in model_para:
            if 'xception' in name:
                if 'bn' in name or 'downsample.1.weight' in name or 'downsample.1.bias' in name:
                    layer_features_BN.append(para)
                else:
                    layer_features.append(para)
            elif 'aspp' in name:
                layer_aspp.append(para)
            elif 'projection' in name:
                layer_projection.append(para)
            elif 'decode' in name:
                layer_decoder.append(para)
            elif 'global' not in name:
                layer_other.append(para)
        return layer_features_BN, layer_features, layer_aspp, layer_projection, layer_decoder, layer_other

    def get_backbone_para(self):
        layer_features = []
        other_features = []
        model_para = list(self.named_parameters())
        for name, para in model_para:
            if 'xception' in name:
                layer_features.append(para)
            else:
                other_features.append(para)
        return layer_features, other_features

    def train_fixbn(self, mode=True, freeze_bn=True, freeze_bn_affine=False):
        super(DeepLabv3_plus_skipconnection_2d, self).train(mode)
        if freeze_bn:
            print("Freezing Mean/Var of BatchNorm2D.")
            if freeze_bn_affine:
                print("Freezing Weight/Bias of BatchNorm2D.")
        if freeze_bn:
            for m in self.xception_features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_state_dict_new(self, state_dict):
        own_state = self.state_dict()
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            new_state_dict[name] = 0
            if name not in own_state:
                if 'num_batch' in name:
                    continue
                print(f'unexpected key "{name}" in state_dict')
                continue
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(f'While copying the parameter named {name}, whose dimensions in the model are'
                      f' {own_state[name].size()} and whose dimensions in the checkpoint are {param.size()}, ...')
                continue
            own_state[name].copy_(param)