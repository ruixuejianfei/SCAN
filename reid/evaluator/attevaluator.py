from __future__ import print_function, absolute_import
import time
import torch
from torch.autograd import Variable
from utils.meters import AverageMeter
from utils import to_numpy
from .eva_functions import cmc, mean_ap
import numpy as np
import torch.nn.functional as F


def evaluate_seq(distmat, query_pids, query_camids, gallery_pids, gallery_camids, cmc_topk=(1, 5, 10, 20)):
    query_ids = np.array(query_pids)
    gallery_ids = np.array(gallery_pids)
    query_cams = np.array(query_camids)
    gallery_cams = np.array(gallery_camids)

    ##
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    top1 = cmc_scores['allshots'][0]
    top5 = cmc_scores['allshots'][4]
    top10 = cmc_scores['allshots'][9]
    top20 = cmc_scores['allshots'][19]

    return mAP, top1, top5, top10, top20



class ATTEvaluator(object):

    def __init__(self, cnn_model, att_model, classifier_model,mode,criterion):
        super(ATTEvaluator, self).__init__()
        self.cnn_model = cnn_model
        self.att_model = att_model
        self.classifier_model = classifier_model
        self.mode = mode
        self.criterion = criterion

    def extract_feature(self, data_loader):
        print_freq = 5
        self.cnn_model.eval()
        self.att_model.eval()


        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        # allfeatures = 0
        # allfeatures_raw = 0

        for i, (imgs, flows, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            imgs = Variable(imgs, volatile=True)
            flows = Variable(flows, volatile=True)

            if i == 0:
                out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)
                allfeatures = [out_feat]
                allfeatures_raw = [out_raw]
                preimgs = imgs
                preflows = flows
            elif imgs.size(0) < data_loader.batch_size:
                flaw_batchsize = imgs.size(0)
                cat_batchsize = data_loader.batch_size - flaw_batchsize
                imgs = torch.cat((imgs, preimgs[0:cat_batchsize]), 0)
                flows = torch.cat((flows, preflows[0:cat_batchsize]), 0)

                out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)

                out_feat = out_feat[0:flaw_batchsize]
                out_raw = out_raw[0:flaw_batchsize]

                allfeatures.append(out_feat)
                allfeatures_raw.append(out_raw)
            else:
                out_feat, out_raw = self.cnn_model(imgs, flows, self.mode)

                allfeatures.append(out_feat)
                allfeatures_raw.append(out_raw)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

        allfeatures = torch.cat(allfeatures, 0)
        allfeatures_raw = torch.cat(allfeatures_raw, 0)
        return allfeatures, allfeatures_raw

    def evaluate(self, query_loader, gallery_loader, queryinfo, galleryinfo):


        self.cnn_model.eval()
        self.att_model.eval()
        self.classifier_model.eval()


        querypid = queryinfo.pid
        querycamid = queryinfo.camid
        querytranum = queryinfo.tranum
        gallerypid = galleryinfo.pid
        gallerycamid = galleryinfo.camid
        gallerytranum = galleryinfo.tranum

        query_resfeatures, query_resraw = self.extract_feature(query_loader)
        gallery_resfeatures, gallery_resraw = self.extract_feature(gallery_loader)

        querylen = len(querypid)
        gallerylen = len(gallerypid)

        # online gallery extraction
        single_distmat = np.zeros((querylen, gallerylen))

        q_start = 0
        pooled_query = []
        for qind, qnum in enumerate(querytranum):
            query_feat_tmp = query_resfeatures[q_start:q_start+qnum, :, :]
            query_featraw_tmp = query_resraw[q_start:q_start+qnum, :, :]
            pooled_query_tmp, hidden_query_tmp = self.att_model.selfpooling_model(query_feat_tmp, query_featraw_tmp)
            pooled_query.append(pooled_query_tmp)
            q_start += qnum
        pooled_query = torch.cat(pooled_query, 0)

        g_start = 0
        pooled_gallery = []
        for gind, gnum in enumerate(gallerytranum):
            gallery_feat_tmp = gallery_resfeatures[g_start:g_start+gnum, :, :]
            gallery_featraw_tmp = gallery_resraw[g_start:g_start+gnum, :, :]
            pooled_gallery_tmp, hidden_gallery_tmp = self.att_model.selfpooling_model(gallery_feat_tmp, gallery_featraw_tmp)
            pooled_gallery.append(pooled_gallery_tmp)
            g_start += gnum
        pooled_gallery = torch.cat(pooled_gallery, 0)
        # pooled_query, hidden_query = self.att_model.selfpooling_model_1(query_resfeatures, query_resraw)
        # pooled_gallery, hidden_gallery = self.att_model.selfpooling_model_2(gallery_resfeatures, gallery_resraw)

        g_start = 0
        pooled_gallery_2 = []
        for gind, gnum in enumerate(gallerytranum):
            gallery_feat_tmp = gallery_resfeatures[g_start:g_start+gnum, :, :]
            gallery_featraw_tmp = gallery_resraw[g_start:g_start+gnum, :, :]
            pooled_gallery_2_tmp = self.att_model.crosspooling_model(gallery_feat_tmp, gallery_featraw_tmp, pooled_query)
            pooled_gallery_2.append(pooled_gallery_2_tmp)
            g_start += gnum
        pooled_gallery_2 = torch.cat(pooled_gallery_2, 1)

        q_start = 0
        pooled_query_2 = []
        for qind, qnum in enumerate(querytranum):
            query_feat_tmp = query_resfeatures[q_start:q_start+qnum, :, :]
            query_featraw_tmp = query_resraw[q_start:q_start+qnum, :, :]
            pooled_query_2_tmp = self.att_model.crosspooling_model(query_feat_tmp, query_featraw_tmp, pooled_gallery)
            pooled_query_2.append(pooled_query_2_tmp)
            q_start += qnum
        pooled_query_2 = torch.cat(pooled_query_2, 1)

        pooled_query_2 = pooled_query_2.permute(1, 0, 2)
        pooled_query, pooled_gallery = pooled_query.unsqueeze(1), pooled_gallery.unsqueeze(0)

        encode_scores = self.classifier_model(pooled_query, pooled_gallery_2, pooled_query_2, pooled_gallery)

        encode_size = encode_scores.size()
        encodemat = encode_scores.view(-1, 2)
        encodemat = F.softmax(encodemat)
        encodemat = encodemat.view(encode_size[0], encode_size[1], 2)

        single_distmat_all = encodemat[:, :, 0]
        single_distmat_all = single_distmat_all.data.cpu().numpy()
        q_start, g_start = 0, 0
        for qind, qnum in enumerate(querytranum):
            for gind, gnum in enumerate(gallerytranum):
                distmat_qg = single_distmat_all[q_start:q_start+qnum, g_start:g_start+gnum]
                #percile = np.percentile(distmat_qg, 20)
                percile = np.percentile(distmat_qg, 10)
                if distmat_qg[distmat_qg <= percile] is not None:
                    distmean = np.mean(distmat_qg[distmat_qg <= percile])
                else:
                    distmean = np.mean(distmat_qg)

                single_distmat[qind, gind] = distmean
                g_start = g_start + gnum
            g_start = 0
            q_start = q_start + qnum

        return evaluate_seq(single_distmat, querypid, querycamid, gallerypid, gallerycamid)
