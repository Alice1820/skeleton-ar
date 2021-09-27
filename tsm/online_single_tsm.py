#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Amirreza Shaban

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: amirreza
"""

import sys
from termcolor import colored
import builtins
import warnings
import shutil
import time
import datetime
# Make sure MFAS is in the path
sys.path.append('mfas')
sys.path.append('distiller_zoo')
# import models.central.ntu as I3D
import tsm as TSM
# from main_found_ntu import *

import torch
import torch.nn as nn
# import torch.autograd.Variable as Variable
import argparse
import os

from tqdm import tqdm
import threading

# distributed
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from k4aBodyTracker import k4aBodyTracker
from zenseTracker import zenseTracker, zenseLoader
import cv2
import numpy as np

#%% Parse inputs
def parse_args():
  # dataset, training settings
  parser = argparse.ArgumentParser(description='Modality optimization.')
  parser.add_argument('--checkpointdir', type=str, help='output base dir', default='checkpoints')
  parser.add_argument('--ckpt_path', type=str, help='output base dir', default='checkpoints/')
  parser.add_argument('--datadir', type=str, help='data directory', default='/data0/xifan/NTU_RGBD_60')
  parser.add_argument('--data', type=str, help='data type', default='fallv3')
  parser.add_argument('--depthmode', type=str, help='data type', default='depthp')
  parser.add_argument('--res_cp', type=str, help='Full net checkpoint (assuming is contained in checkpointdir)', default='')
  parser.add_argument('--num_outputs', type=int, help='output dimension', default=60)
  parser.add_argument('--num_class', type=int, help='', default=120)
  parser.add_argument('--batchsize', type=int, help='batch size', default=20)
  parser.add_argument('--epochs', type=int, help='training epochs', default=700)
  parser.add_argument('--use_dataparallel', help='Use several GPUs', action='store_true', dest='use_dataparallel', default=True)
  parser.add_argument('--j', dest='num_workers', type=int, help='Dataloader CPUS', default=32)
  parser.add_argument('--modality', type=str, help='', default='RGB')
  parser.add_argument('--task', type=str, help='', default='none')
  parser.add_argument('--no-multitask', dest='multitask', help='Multitask loss', action='store_false', default=True)

  parser.add_argument("--vid_len", action="store", default=(8, 32), dest="vid_len", type=int, nargs='+', help="length of video, as a tuple of two lengths, (rgb len, skel len)")
  parser.add_argument("--drpt", action="store", default=0.4, dest="drpt", type=float, help="dropout")

  parser.add_argument('--no_bad_skel', action="store_true", help='Remove the 300 bad samples, espec. useful to evaluate', default=False)
  parser.add_argument("--no_norm", action="store_true", default=False, dest="no_norm", help="Not normalizing the skeleton")

  parser.add_argument('--train', action='store_true', default=False, help='training')
  parser.add_argument('--mode', type=str, help='NTU mode', default='cross_view')

  parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
  parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                      dest='weight_decay')
  # depth augmentation
  parser.add_argument('--aug', type=str, default='none', help='depth augmentation (default:none): aug1, aug2, rgb')

  # distributed
  parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
                      help='path to latest checkpoint (default: none)')
  parser.add_argument('--world-size', default=1, type=int,
                      help='number of nodes for distributed training')
  parser.add_argument('--rank', default=0, type=int,
                      help='node rank for distributed training')
  parser.add_argument('--dist-url', default='tcp://localhost:10041', type=str,
                      help='url used to set up distributed training')
  parser.add_argument('--dist-backend', default='nccl', type=str,
                      help='distributed backend')
  parser.add_argument('--seed', default=None, type=int,
                      help='seed for initializing training. ')
  parser.add_argument('--gpu', default=None, type=int,
                      help='GPU id to use.')
  parser.add_argument('--multiprocessing-distributed', action='store_true',
                      help='Use multi-processing distributed training to launch '
                          'N processes per node, which has N GPUs. This is the '
                          'fastest way to use PyTorch for either single node or '
                          'multi node data parallel training')

  # tsm
  parser.add_argument('--arch', type=str, default='resnet18', help='')
  parser.add_argument('--num_segments', type=int, help='', default=8)
  parser.add_argument('--img_feature_dim', type=int, help='', default=256)
  parser.add_argument("--dropout", action="store", default=0.8, type=float, help="dropout")
  parser.add_argument('--consensus_type', type=str, default='avg', help='')
  parser.add_argument('--shift', action='store_true', default=False, help='')
  parser.add_argument('--shift_div', type=int, help='', default=8)
  parser.add_argument('--shift_place', type=str, default='blockres', help='')
  parser.add_argument("--lr_steps", action="store", default=(10, 20), dest="vid_len", type=int, nargs='+', help="length of video, as a tuple of two lengths, (rgb len, skel len)")
  parser.add_argument("--non_local", action="store_true", default=False, dest="non_local", help="") # add non local module

  return parser.parse_args()

def update_lr(optimizer, multiplier = .1):
  state_dict = optimizer.state_dict()
  for param_group in state_dict['param_groups']:
    param_group['lr'] = param_group['lr'] * multiplier
  optimizer.load_state_dict(state_dict)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def demo_step_track(model, datatracker, args):
  # time.sleep(5)
  model.train(False)
  # losses = AverageMeter('Loss', ':.4e')
  # top1 = AverageMeter('Acc@1', ':6.2f')
  # top5 = AverageMeter('Acc@5', ':6.2f')
  # progress = ProgressMeter(
  #   len(dataloaders[phase]),
  #   [batch_time, data_time, losses, top1, top5],
  #   prefix="Epoch: [{}]".format(epoch))
  # Iterate over data
  # for data in tqdm(dataloaders['test']):
  #   visa, visb, label = [data[n].cuda() for n in ['samplea', 'sampleb', 'label']]
  #   with torch.set_grad_enabled(False):
  #     output, target = model(im_q=visa, im_k=visb)
  #     loss = criterion(output, target) # 
  #     # print (loss.size())
  #   batch_size = visa.size(0)
  #   acc1, acc5 = accuracy(output, target, topk=(1, 5))
  #   losses.update(loss.item(), batch_size)
  #   top1.update(acc1[0], batch_size)
  #   top5.update(acc5[0], batch_size)
  # print
  # print('Loss I3D {:.6f}'.format(losses.avg))
  # print('Acc I3D Top1: {:.6f}, Top5: {:.6f}'.format(top1.avg, top5.avg))
  # return top1.avg
  counts_f = 0
  start_time = time.time()
  while True:
    dep = datatracker.next_clip()
    _cmap = datatracker._cmapNow
    ts = datatracker.ts
    if dep is not None:
      # print (dep.size()) # [bs*8, 3, 224, 224]
      # model.load_state_dict(torch.load(args.ckpt_path))
      # print (torch.min(dep), torch.mean(dep), torch.max(dep))
      with torch.set_grad_enabled(False):
        feat = model(dep)
        out = F.softmax(feat, -1)
        probs = out[0].cpu().numpy()
        prob = torch.max(out[0]).cpu().numpy()
        pred = torch.argmax(out[0]).cpu().numpy()
        # print out result
        # if prob > 0.5:
        # print (torch.mean(dep, (1, 2, 3)).cpu().numpy())
        out = out.cpu().numpy()
        # ind = np.argpartition(out[0], -5)[-5:]
        # print (datetime.datetime.now().__str__() + '         ', '{:.2f}%'.format(prob * 1e2), '{:03d}'.format(pred + 1), \
            # idx2label[args.data][int(pred) + 1])
        # if pred == 1 and prob >= 0.8:
        iter_time = time.time() - start_time
        time.sleep(0.33)
        if pred == 1:
          print (datetime.datetime.now().__str__() + '         ', \
              colored(idx2label[args.data][int(pred) + 1], 'red'), '{:.2f}%'.format(prob * 1e2), 'Iter: {:.2f}ms'.format(iter_time * 1000))  
          # os.system("beep -f 555 -l 460")
          # falling
          counts_f += 1
          for idx, cmap in enumerate(_cmap):
            cv2.imwrite('output/{:02d}/falling{:02d}.jpg'.format(counts_f, idx), cmap)
          # print (ts)
          # exit()
        else:
          print (datetime.datetime.now().__str__() + '         ', \
          colored('normal', 'green'), '{:.2f}%'.format(probs[1] * 1e2), 'Iter: {:.2f}ms'.format(iter_time * 1000))
            # [idx2label[args.data][ind[i] + 1] for i in range(5)]
        start_time = time.time()
        
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main_worker(gpu, ngpus_per_node, args):
  args.gpu = gpu
  print("=> creating model")
  model = TSM.TSN(num_class=args.num_class,
                  num_segments=args.num_segments,
                  modality=args.modality,
                  base_model=args.arch,
                  # consensus_type=args.consensus_type,
                  # dropout=args.dropout,
                  # img_feature_dim=args.img_feature_dim,
                  # partial_bn=not args.no_partialbn,
                  # pretrain=args.pretrain,
                  # is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                  # fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                  # temporal_pool=args.temporal_pool,
                  # non_local=args.non_local
                  )
  # print(model)
  # criterion = torch.nn.CrossEntropyLoss()
  # optimizer = torch.optim.SGD(model.parameters(), 1e-4,
  #                             momentum=args.momentum,
  #                             weight_decay=args.weight_decay)

  # suppress printing if not master
  # if args.multiprocessing_distributed and args.gpu != 0:
      # def print_pass(*args):
      #     pass
      # builtins.print = print_pass
  # if args.distributed:
  #   # if args.dist_url == "env://" and args.rank == -1:
  #   #     args.rank = int(os.environ["RANK"])
  #   if args.multiprocessing_distributed:
  #       # For multiprocessing distributed training, rank needs to be the
  #       # global rank among all the processes
  #       # args.rank = args.rank * ngpus_per_node + gpu
  #       args.rank = args.rank * ngpus_per_node
  #   dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
  #                           world_size=args.world_size, rank=args.rank)
  model.cuda()
  model = torch.nn.DataParallel(model)
  if args.ckpt_path is not None:
    try:
      model.load_state_dict(torch.load(args.ckpt_path))
    except:
      model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
    print('Successfully loaded pretrained model' + args.ckpt_path)
  else:
    raise Exception("pretrained checkpoint is None.")
  # DistributedDataParallel will divide and allocate batch_size to all
  # available GPUs if device_ids are not set
  # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
  epoch = 0
  # if args.res_cp is not '':
  #   # load whole model
  #   model_filename = os.path.join(args.checkpointdir, args.res_cp)
  #   # print('Loaded ' + model_filename)
  #   model.load_state_dict(torch.load(model_filename)['state_dict'])
  #   optimizer.load_state_dict(torch.load(model_filename)['optimizer'])
  #   for state in optimizer.state.values():
  #     for k, v in state.items():
  #         if torch.is_tensor(v):
  #             state[k] = v.cuda()
  #   epoch = torch.load(model_filename)['epoch']
  #   # model.cnn.load_state_dict(moco.encoder_k.state_dict)
  #   print('Loaded ' + model_filename)
  # if args.res_cp is not '':
  #   model_filename = os.path.join(args.checkpointdir, args.stu_cp)
  #   model.load_state_dict(torch.load(model_filename))
  #   print('Loaded ' + model_filename)

  # datasets = {'none': get_datasets_ntudep(args),
              # 'rgb': get_datasets_nturgb(args),
              # 'aug1': get_datasets_ntuaug(args),
              # }[args.aug]
  #
  # datasets = get_datasets_ntudep(args)
  # datasets = get_datasets_arrowdepv2(args)
  # dataloaders = {}
  # if args.distributed:
  #     for x in ['train', 'trains', 'test']:
  #       sampler = torch.utils.data.distributed.DistributedSampler(datasets[x])
  # else:
  #     sampler = None
  # for x in ['train', 'test']:
  #   dataloaders[x] = torch.utils.data.DataLoader(
  #       datasets[x], batch_size=args.batchsize, shuffle=(x == 'train'),
  #       num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=True)
  # if args.train:
  #   train_step_track_acc(model, criterion, optimizer, dataloaders, res_epochs=epoch)
  # else:
  #   test_step_track_acc(model, criterion, optimizer, dataloaders)
  print ('model initialized')
  # datatracker = k4aBodyTracker()
  # datatracker = zenseTracker(args)
  datatracker = zenseLoader(args)
  threading.Thread(target=demo_step_track, args=(model, datatracker, args, ), daemon=True).start()
  k = 0
  while True:
    # if datatracker.imageNow is not None:
    if datatracker.cmapImgNow is not None:
      # Overlay body segmentation on depth image
      cv2.imshow('Segmented Depth Image', datatracker.imageNow)
      # cv2.imshow('Segmented Depth Image', datatracker.cmapImgNow)
      k = cv2.waitKey(1)
    if k == 27:
      break
    elif k == ord('q'):
      cv2.imwrite('outputImage.jpg', datatracker.imageNow)
      cv2.imwrite('outputDepth.png', datatracker.depthNow)
       
if __name__ == "__main__":
  print("Training TSM Network Distributed")
  args = parse_args()
  # args.ckpt_path = os.path.join(args.ckpt_path, 'tsm_{}_mobilenetv2_arrowv3.pth.tar'.format(args.depthmode))
  # print("The configuration of this run is:")

  use_gpu = torch.cuda.is_available()
  print (use_gpu)
  # device = torch.device("cuda:0" if use_gpu else "cpu")
  # if args.dist_url == "env://" and args.world_size == -1:
  #       args.world_size = int(os.environ["WORLD_SIZE"])

  # args.distributed = args.world_size > 1 or args.multiprocessing_distributed

  ngpus_per_node = 0
  # ngpus_per_node = torch.cuda.device_count()
  # ngpus_per_node = 1
  # if args.multiprocessing_distributed:
  #     # Since we have ngpus_per_node processes per node, the total world_size
  #     # needs to be adjusted accordingly
  #     args.world_size = ngpus_per_node * args.world_size
  #     # Use torch.multiprocessing.spawn to launch distributed processes: the
  #     # main_worker process function
  #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
  # else:
  #     # Simply call main_worker function
  main_worker(args.gpu, ngpus_per_node, args)