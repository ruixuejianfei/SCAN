# system tool
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
# computation tool
import torch
import numpy as np
# device tool
import torch.backends.cudnn as cudnn
# utilis
from utils.logging import Logger
from reid import models
from utils.serialization import load_checkpoint, save_cnn_checkpoint, save_att_checkpoint, save_cls_checkpoint
from reid.loss import PairLoss, OIMLoss
from reid.data import get_data
from reid.train import SEQTrainer
from reid.evaluator import CNNEvaluator
from reid.evaluator import ATTEvaluator
from tensorboardX import SummaryWriter


def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # log file

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    dataset, num_classes, train_loader, query_loader, gallery_loader = \
        get_data(args.dataset, args.split, args.data_dir,
                 args.batch_size, args.seq_len, args.seq_srd,
                 args.workers, args.train_mode)

    # create CNN model
    cnn_model = models.create(args.a1, num_features=args.features, dropout=args.dropout)

    # create ATT model
    input_num = cnn_model.feat.in_features
    output_num = args.features
    att_model = models.create(args.a2, input_num, output_num)

    # create classifier model
    class_num = 2
    classifier_model = models.create(args.a3,  output_num, class_num)


    # CUDA acceleration model

    cnn_model = torch.nn.DataParallel(cnn_model).cuda()
    att_model = att_model.cuda()
    classifier_model = classifier_model.cuda()


    # Loss function

    criterion_oim = OIMLoss(args.features, num_classes,
                            scalar=args.oim_scalar, momentum=args.oim_momentum)
    criterion_veri = PairLoss(args.sampling_rate)
    criterion_oim.cuda()
    criterion_veri.cuda()

    # Optimizer
    base_param_ids = set(map(id, cnn_model.module.base.parameters()))
    new_params = [p for p in cnn_model.parameters() if
                  id(p) not in base_param_ids]

    param_groups1 = [
        {'params': cnn_model.module.base.parameters(), 'lr_mult': 1},
        {'params': new_params, 'lr_mult': 1}]
    param_groups2 = [
        {'params': att_model.parameters(), 'lr_mult': 1},
        {'params': classifier_model.parameters(), 'lr_mult': 1}]




    optimizer1 = torch.optim.SGD(param_groups1, lr=args.lr1,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)


    optimizer2 = torch.optim.SGD(param_groups2, lr=args.lr2,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay,
                                 nesterov=True)




    # Schedule Learning rate
    def adjust_lr1(epoch):
        lr = args.lr1 * (0.1 ** (epoch/args.lr1step))
        print(lr)
        for g in optimizer1.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    def adjust_lr2(epoch):
        lr = args.lr2 * (0.01 ** (epoch//args.lr2step))
        print(lr)
        for g in optimizer2.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    def adjust_lr3(epoch):
        lr = args.lr3 * (0.000001 ** (epoch //args.lr3step))
        print(lr)
        return lr


    best_top1 = 0
    start_epoch = args.start_epoch
    if args.evaluate == 1:
        print('Evaluate:')
        evaluator = ATTEvaluator(cnn_model, att_model, classifier_model, args.train_mode, criterion_veri)
        top1, mAP = evaluator.evaluate(query_loader, gallery_loader, dataset.queryinfo, dataset.galleryinfo)

    elif args.test == 1:
        print('Test:')
        checkpoint1 = load_checkpoint(osp.join(args.logs_dir, 'cnnmodel_best.pth.tar'))
        cnn_model.load_state_dict(checkpoint1['state_dict'])
        checkpoint2 = load_checkpoint(osp.join(args.logs_dir, 'attmodel_best.pth.tar'))
        att_model.load_state_dict(checkpoint2['state_dict'])
        checkpoint3 = load_checkpoint(osp.join(args.logs_dir, 'clsmodel_best.pth.tar'))
        classifier_model.load_state_dict(checkpoint3['state_dict'])
        evaluator = ATTEvaluator(cnn_model, att_model, classifier_model, args.train_mode, criterion_veri)
        mAP, top1, top5, top10, top20 = evaluator.evaluate(query_loader, gallery_loader, dataset.queryinfo, dataset.galleryinfo)

    else:
        tensorboard_test_logdir = osp.join(args.logs_dir, 'test_log')
        writer = SummaryWriter(log_dir=tensorboard_test_logdir)
        if args.resume == 1:
            checkpoint1 = load_checkpoint(osp.join(args.logs_dir, 'cnn_checkpoint.pth.tar'))
            cnn_model.load_state_dict(checkpoint1['state_dict'])
            checkpoint2 = load_checkpoint(osp.join(args.logs_dir, 'att_checkpoint.pth.tar'))
            att_model.load_state_dict(checkpoint2['state_dict'])
            checkpoint3 = load_checkpoint(osp.join(args.logs_dir, 'cls_checkpoint.pth.tar'))
            classifier_model.load_state_dict(checkpoint3['state_dict'])
            start_epoch = checkpoint1['epoch']
            best_top1 = checkpoint1['best_top1']
            print("=> Start epoch {}  best top1 {:.1%}"
                  .format(start_epoch, best_top1))
        # Trainer
        tensorboard_train_logdir = osp.join(args.logs_dir, 'train_log')
        trainer = SEQTrainer(cnn_model, att_model, classifier_model, criterion_veri, criterion_oim, args.train_mode, args.lr3, tensorboard_train_logdir)
        # Evaluator
        if args.train_mode == 'cnn':
            evaluator = CNNEvaluator(cnn_model, args.train_mode)
        elif args.train_mode == 'cnn_rnn':
            evaluator = ATTEvaluator(cnn_model, att_model, classifier_model, args.train_mode, criterion_veri)
        else:
            raise RuntimeError('Yes, Evaluator is necessary')

        for epoch in range(start_epoch, args.epochs):
            adjust_lr1(epoch)
            adjust_lr2(epoch)
            rate = adjust_lr3(epoch)
            trainer.train(epoch, train_loader, optimizer1, optimizer2, rate)

            if epoch % 1 == 0:
                mAP, top1, top5, top10, top20 = evaluator.evaluate(query_loader, gallery_loader, dataset.queryinfo, dataset.galleryinfo)
                writer.add_scalar('test/mAP', mAP, epoch+1)
                writer.add_scalar('test/top1', top1, epoch+1)
                writer.add_scalar('test/top5', top5, epoch+1)
                writer.add_scalar('test/top10', top10, epoch+1)
                writer.add_scalar('test/top20', top20, epoch+1)
                is_best = top1 > best_top1
                if is_best:
                    best_top1 = top1

                save_cnn_checkpoint({
                    'state_dict': cnn_model.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(args.logs_dir, 'cnn_checkpoint.pth.tar'))

                if args.train_mode == 'cnn_rnn':
                    save_att_checkpoint({
                        'state_dict': att_model.state_dict(),
                        'epoch': epoch + 1,
                        'best_top1': best_top1,
                    }, is_best, fpath=osp.join(args.logs_dir, 'att_checkpoint.pth.tar'))

                    save_cls_checkpoint({
                        'state_dict': classifier_model.state_dict(),
                        'epoch': epoch + 1,
                        'best_top1': best_top1,
                    }, is_best, fpath=osp.join(args.logs_dir, 'cls_checkpoint.pth.tar'))

        print('Test: ')
        checkpoint1 = load_checkpoint(osp.join(args.logs_dir, 'cnnmodel_best.pth.tar'))
        cnn_model.load_state_dict(checkpoint1['state_dict'])
        checkpoint2 = load_checkpoint(osp.join(args.logs_dir, 'attmodel_best.pth.tar'))
        att_model.load_state_dict(checkpoint2['state_dict'])
        checkpoint3 = load_checkpoint(osp.join(args.logs_dir, 'clsmodel_best.pth.tar'))
        classifier_model.load_state_dict(checkpoint3['state_dict'])
        evaluator = ATTEvaluator(cnn_model, att_model, classifier_model, args.train_mode, criterion_veri)
        mAP, top1, top5, top10, top20 = evaluator.evaluate(query_loader, gallery_loader, dataset.queryinfo, dataset.galleryinfo)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ID Training ResNet Model")

    # DATA
    parser.add_argument('-d', '--dataset', type=str, default='prid2011sequence',
                        choices=['marssequence', 'ilidsvidsequence', 'prid2011sequence'])
    parser.add_argument('-b', '--batch-size', type=int, default=8)

    parser.add_argument('-j', '--workers', type=int, default=4)

    parser.add_argument('--seq_len', type=int, default=10)

    parser.add_argument('--seq_srd', type=int, default=5)

    parser.add_argument('--split', type=int, default=0)

    # MODEL
    # CNN model
    parser.add_argument('--a1', '--arch_1', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)


    # Attention model
    parser.add_argument('--a2', '--arch_2', type=str, default='attmodel',
                        choices=models.names())


    # Classifier_model
    parser.add_argument('--a3', '--arch_3', type=str, default='classifier',
                        choices=models.names())

    # Criterion model
    parser.add_argument('--loss', type=str, default='oim',
                        choices=['xentropy', 'oim', 'triplet'])
    parser.add_argument('--oim-scalar', type=float, default=30)
    parser.add_argument('--oim-momentum', type=float, default=0.5)
    parser.add_argument('--sampling-rate', type=int, default=3)

    # OPTIMIZER
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr1', type=float, default=0.001)
    parser.add_argument('--lr2', type=float, default=0.001)
    parser.add_argument('--lr3', type=float, default=0.001)

    parser.add_argument('--lr1step', type=float, default=10)
    parser.add_argument('--lr2step', type=float, default=10)
    parser.add_argument('--lr3step', type=float, default=10)

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--cnn_resume', type=str, default='', metavar='PATH')

    # TRAINER
    parser.add_argument('--train_mode', type=str, default='cnn_rnn',
                        choices=['cnn_rnn', 'cnn'])
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--evaluate', type=int, default=0)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--resume', type=int, default=0)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    # main function
    main(parser.parse_args())
