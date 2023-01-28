import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.333)
        #nn.init.normal_(m.weight, mean=0.3, std=0.1)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        pool_dim = 2048
        self.l2norm = Normalize(2)        
        self.bottleneck41 = nn.BatchNorm1d(pool_dim)
        self.bottleneck41.bias.requires_grad_(False)  # no shift
        self.bottleneck41.apply(weights_init_kaiming)
        self.classifier41 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier41.apply(weights_init_classifier)
        
        self.bottleneck42 = nn.BatchNorm1d(pool_dim)
        self.bottleneck42.bias.requires_grad_(False)  # no shift
        self.bottleneck42.apply(weights_init_kaiming)
        self.classifier42 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier42.apply(weights_init_classifier)
        
        self.bottleneck43 = nn.BatchNorm1d(pool_dim)
        self.bottleneck43.bias.requires_grad_(False)  # no shift
        self.bottleneck43.apply(weights_init_kaiming)
        self.classifier43 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier43.apply(weights_init_classifier)
        
        self.bottleneck44 = nn.BatchNorm1d(pool_dim)
        self.bottleneck44.bias.requires_grad_(False)  # no shift
        self.bottleneck44.apply(weights_init_kaiming)
        self.classifier44 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier44.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)

            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)


        # shared block
        x = self.base_resnet.base.layer1(x)
        x = self.base_resnet.base.layer2(x)
        x = self.base_resnet.base.layer3(x)
        x = self.base_resnet.base.layer4(x)
        x41, x42, x43, x44 = torch.chunk(x, 4, 2)
        
        x41 = self.avgpool(x41)
        x42 = self.avgpool(x42)
        x43 = self.avgpool(x43)
        x44 = self.avgpool(x44)
        x41 = x41.view(x41.size(0), x41.size(1))
        x42 = x42.view(x42.size(0), x42.size(1))
        x43 = x43.view(x43.size(0), x43.size(1))
        x44 = x44.view(x44.size(0), x44.size(1))

        feat41 = self.bottleneck41(x41)
        feat42 = self.bottleneck42(x42)
        feat43 = self.bottleneck43(x43)
        feat44 = self.bottleneck44(x44)

        if self.training:
            # return x41, x42, x43, x44, self.classifier41(feat41), self.classifier42(feat42), self.classifier43(feat43), self.classifier44(feat44), [xo, gray]
            return x41, x42, x43, x44, self.classifier41(feat41), self.classifier42(feat42), self.classifier43(feat43), self.classifier44(feat44)
        else:
            return self.l2norm(torch.cat((x41, x42, x43, x44),1)), self.l2norm(torch.cat((feat41, feat42, feat43, feat44),1))


class embed_net_ctn(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net_ctn, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        pool_dim = 2048
        self.l2norm = Normalize(2)        
        self.bottleneck41 = nn.BatchNorm1d(pool_dim)
        self.bottleneck41.bias.requires_grad_(False)  # no shift
        self.bottleneck41.apply(weights_init_kaiming)
        self.classifier41 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier41.apply(weights_init_classifier)
        
        self.bottleneck42 = nn.BatchNorm1d(pool_dim)
        self.bottleneck42.bias.requires_grad_(False)  # no shift
        self.bottleneck42.apply(weights_init_kaiming)
        self.classifier42 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier42.apply(weights_init_classifier)
        
        self.bottleneck43 = nn.BatchNorm1d(pool_dim)
        self.bottleneck43.bias.requires_grad_(False)  # no shift
        self.bottleneck43.apply(weights_init_kaiming)
        self.classifier43 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier43.apply(weights_init_classifier)
        
        self.bottleneck44 = nn.BatchNorm1d(pool_dim)
        self.bottleneck44.bias.requires_grad_(False)  # no shift
        self.bottleneck44.apply(weights_init_kaiming)
        self.classifier44 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier44.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.head41 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128)
            )

        self.head42 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128)
            )

        self.head43 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128)
            )

        self.head44 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128)
            )

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)

            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)


        # shared block
        x = self.base_resnet.base.layer1(x)
        x = self.base_resnet.base.layer2(x)
        x = self.base_resnet.base.layer3(x)
        x = self.base_resnet.base.layer4(x)
        x41, x42, x43, x44 = torch.chunk(x, 4, 2)
        
        x41 = self.avgpool(x41)
        x42 = self.avgpool(x42)
        x43 = self.avgpool(x43)
        x44 = self.avgpool(x44)
        x41 = x41.view(x41.size(0), x41.size(1))
        x42 = x42.view(x42.size(0), x42.size(1))
        x43 = x43.view(x43.size(0), x43.size(1))
        x44 = x44.view(x44.size(0), x44.size(1))

        feat41 = self.bottleneck41(x41)
        feat42 = self.bottleneck42(x42)
        feat43 = self.bottleneck43(x43)
        feat44 = self.bottleneck44(x44)

        feat_head41 = self.l2norm(self.head41(feat41))
        feat_head42 = self.l2norm(self.head41(feat42))
        feat_head43 = self.l2norm(self.head41(feat43))
        feat_head44 = self.l2norm(self.head41(feat44))



        if self.training:
            # return x41, x42, x43, x44, self.classifier41(feat41), self.classifier42(feat42), self.classifier43(feat43), self.classifier44(feat44), [xo, gray]
            return feat_head41, feat_head42, feat_head43, feat_head44, x41, x42, x43, x44, self.classifier41(feat41), self.classifier42(feat42), self.classifier43(feat43), self.classifier44(feat44)
        else:
            return self.l2norm(torch.cat((x41, x42, x43, x44),1)), self.l2norm(torch.cat((feat41, feat42, feat43, feat44),1))
            
            
            
            
            
            
            
            
            
            