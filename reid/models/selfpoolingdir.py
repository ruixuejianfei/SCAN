from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.init as init

class SelfPoolingDir(nn.Module):
    def __init__(self, input_num, output_num, feat_fc=None):  # 2048,128
        super(SelfPoolingDir, self).__init__()
        self.input_num = input_num
        self.output_num = output_num

        ## Linear_Q
        if feat_fc is None:
            self.featQ = nn.Sequential(nn.Linear(self.input_num, self.output_num),
                                       nn.BatchNorm1d(self.output_num))
            for m in self.featQ.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal(m.weight, mode='fan_out')
                    init.constant(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    init.constant(m.weight, 1)
                    init.constant(m.bias, 0)
                else:
                    print(type(m))
        else:
            self.featQ = feat_fc

        ## Softmax
        self.softmax = nn.Softmax()

    def forward(self, probe_value, probe_base): #(bz/2)*sq*128; (bz/2)*sq*2048
        pro_size = probe_value.size()
        pro_batch = pro_size[0]  # 32
        pro_len = pro_size[1]    # 10

        # generating Querys
        Qs = probe_base.view(pro_batch * pro_len, -1)  # 320*2048
        Qs = self.featQ(Qs)
        # Qs = self.featQ_bn(Qs)
        Qs = Qs.view(pro_batch, pro_len, -1)  # 32*10*128
        tmp_K = Qs
        Qmean = torch.mean(Qs, 1, keepdim=True)  # 32*1*128
        Hs = Qmean.expand(pro_batch, pro_len, self.output_num)  # 32*10*128

        weights = Hs * tmp_K                    # 32*10*128
        weights = weights.permute(0, 2, 1)  # 32*128*10
        weights = weights.contiguous()
        weights = weights.view(-1, pro_len)
        weights = self.softmax(weights)
        weights = weights.view(pro_batch, self.output_num, pro_len)
        weights = weights.permute(0, 2, 1)  # 32*10*128
        pool_probe = probe_value * weights
        pool_probe = pool_probe.sum(1)
        pool_probe = pool_probe.squeeze(1)  # 32*128
        """
        pool_probe = torch.mean(probe_value,1)
        pool_probe = pool_probe.squeeze(1) # 32*128
        """

        # pool_probe  Batch x featnum
        # Hs  Batch x hidden_num

        return pool_probe, pool_probe
