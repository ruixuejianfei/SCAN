from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.init as init
import numpy as np


class Classifier(nn.Module):
    def __init__(self, feat_num, class_num, drop=0):
        super(Classifier, self).__init__()
        self.feat_num = feat_num
        self.class_num = class_num
        self.drop = drop

        # BN layer
        self.classifierBN = nn.BatchNorm1d(self.feat_num)
        # feat classifeir
        self.classifierlinear = nn.Linear(self.feat_num, self.class_num)
        # dropout_layer
        self.drop = drop
        if self.drop > 0:
            self.droplayer = nn.Dropout(drop)

        init.constant(self.classifierBN.weight, 1)
        init.constant(self.classifierBN.bias, 0)

        init.normal(self.classifierlinear.weight, std=0.001)
        init.constant(self.classifierlinear.bias, 0)

    def forward(self, probe, gallery2, probe2, gallery):
        S_gallery2 = gallery2.size()
        N_probe = S_gallery2[0]
        N_gallery = S_gallery2[1]
        feat_num = S_gallery2[2]

        probe = probe.expand(N_probe, N_gallery, feat_num)
        gallery = gallery.expand(N_probe, N_gallery, feat_num)


        slice0 = 30
        if N_probe < slice0:
            diff1, diff2 = probe - gallery, probe2 - gallery2
            diff = diff1 * diff2
            pg_size = diff.size()
            p_size, g_size = pg_size[0], pg_size[1]
            diff = diff.view(p_size * g_size, -1)
            diff = diff.contiguous()
            diff = self.classifierBN(diff)
            if self.drop > 0:
                diff = self.droplayer(diff)
            cls_encode = self.classifierlinear(diff)
            cls_encode = cls_encode.view(p_size, g_size, -1)

        else:
            iter_time_0 = int(np.floor(N_probe / slice0))
            for i in range(0, iter_time_0):
                before_index_0 = i * slice0
                after_index_0 = (i + 1) * slice0
                probe_tmp = probe[before_index_0:after_index_0, :, :]
                gallery_tmp = gallery[before_index_0:after_index_0, :, :]
                probe2_tmp = probe2[before_index_0:after_index_0, :, :]
                gallery2_tmp = gallery2[before_index_0:after_index_0, :, :]
                diff1_tmp, diff2_tmp = probe_tmp - gallery_tmp, probe2_tmp - gallery2_tmp
                # diff1_tmp = diff1[before_index_0:after_index_0, :, :]
                # diff2_tmp = diff2[before_index_0:after_index_0, :, :]
                diff_tmp = diff1_tmp * diff2_tmp
                pg_size = diff_tmp.size()
                p_size, g_size = pg_size[0], pg_size[1]
                diff_tmp = diff_tmp.view(p_size * g_size, -1)
                diff_tmp = diff_tmp.contiguous()
                diff_tmp = self.classifierBN(diff_tmp)
                if self.drop > 0:
                    diff_tmp = self.droplayer(diff_tmp)
                cls_encode_tmp = self.classifierlinear(diff_tmp)
                cls_encode_tmp = cls_encode_tmp.view(p_size, g_size, -1)
                if i == 0:
                    cls_encode = cls_encode_tmp
                else:
                    cls_encode = torch.cat((cls_encode, cls_encode_tmp), 0)
            before_index_0 = iter_time_0 * slice0
            after_index_0 = N_probe
            if after_index_0 > before_index_0:
                probe_tmp = probe[before_index_0:after_index_0, :, :]
                gallery_tmp = gallery[before_index_0:after_index_0, :, :]
                probe2_tmp = probe2[before_index_0:after_index_0, :, :]
                gallery2_tmp = gallery2[before_index_0:after_index_0, :, :]
                diff1_tmp, diff2_tmp = probe_tmp - gallery_tmp, probe2_tmp - gallery2_tmp
                # diff1_tmp = diff1[before_index_0:after_index_0, :, :]
                # diff2_tmp = diff2[before_index_0:after_index_0, :, :]
                diff_tmp = diff1_tmp * diff2_tmp
                pg_size = diff_tmp.size()
                p_size, g_size = pg_size[0], pg_size[1]
                diff_tmp = diff_tmp.view(p_size * g_size, -1)
                diff_tmp = diff_tmp.contiguous()
                diff_tmp = self.classifierBN(diff_tmp)
                if self.drop > 0:
                    diff_tmp = self.droplayer(diff_tmp)
                cls_encode_tmp = self.classifierlinear(diff_tmp)
                cls_encode_tmp = cls_encode_tmp.view(p_size, g_size, -1)
                cls_encode = torch.cat((cls_encode, cls_encode_tmp), 0)

        return cls_encode
