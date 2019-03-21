from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

class AlexNet(nn.Module):
    def __init__(self, pretrained=True, cut_at_pooling=False, num_features=0, dropout=0):
        super(AlexNet, self).__init__()

        self.pretrained = pretrained
        self.has_embedding = num_features > 0
        self.num_features = num_features

        self.base = torchvision.models.alexnet(pretrained=pretrained)
        self.features = self.base.features
        self.classifier = self.base.classifier
        conv0 = nn.Conv2d(2, 64, kernel_size=11, stride=4, padding=2, bias=False)
        init.kaiming_normal(conv0.weight, mode='fan_out')
        self.conv0 = conv0

        out_planes = self.classifier._modules['1'].in_features

        self.feat1 = nn.Linear(5376, 2048)
        self.feat_bn1 = nn.BatchNorm1d(2048)
        init.kaiming_normal(self.feat1.weight, mode='fan_out')
        init.constant(self.feat1.bias, 0)
        init.constant(self.feat_bn1.weight, 1)
        init.constant(self.feat_bn1.bias, 0)

        if self.has_embedding:
            self.feat = nn.Linear(2048, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal(self.feat.weight, mode='fan_out')
            init.constant(self.feat.bias, 0)
            init.constant(self.feat_bn.weight, 1)
            init.constant(self.feat_bn.bias, 0)


    def forward(self, imgs, motions, mode):
        img_size = imgs.size()
        motion_size = motions.size()
        batch_sz = img_size[0]
        seq_len = img_size[1]
        imgs = imgs.view(-1, img_size[2], img_size[3], img_size[4])
        motions = motions.view(-1, motion_size[2], motion_size[3], motion_size[4])
        motions = motions[:, 1:3]

        for name, module in self.features._modules.items():
            if name == '0':
                x = module(imgs) + self.conv0(motions)
                continue
            x = module(x)

        x = x.view(x.size(0), -1)
        x = self.feat1(x)
        x = self.feat_bn1(x)
        if mode == 'cnn_rnn':
            raw = x.view(batch_sz, seq_len, -1)
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)

        if mode == 'cnn_rnn':
            # x = x / x.norm(2, 1).expand_as(x)
            x = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
            x = x.view(batch_sz, seq_len, -1)
            return x, raw

def alexnet(**kwargs):
        return AlexNet(**kwargs)
