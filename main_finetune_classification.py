# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import math
import numpy as np
import os
import pandas as pd
import sys
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from sklearn.metrics import confusion_matrix
import wandb

from lavila.data import datasets
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from lavila.models import models
from lavila.models.tokenizer import (MyBertTokenizer, MyDistilBertTokenizer, MyGPT2Tokenizer, SimpleTokenizer)
from lavila.models.utils import inflate_positional_embeds
from lavila.utils import distributed as dist_utils
from lavila.utils.evaluation import accuracy, get_mean_accuracy
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils.preprocess import generate_label_map
from lavila.utils.random import random_seed
from lavila.utils.scheduler import cosine_scheduler
from lavila.utils.evaluation_ek100cls import get_marginal_indexes, marginalize


def get_args_parser():
    parser = argparse.ArgumentParser(description='lavila finetune and evaluation', add_help=False)
    # Data
    parser.add_argument('--dataset', default='ek100_cls', type=str,
                        choices=['ek100_cls', 'egtea'])
    parser.add_argument('--root',
                        default='datasets/EK100/video_ht256px/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata-train',
                        default='datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv',
                        type=str, help='path to metadata file (train set)')
    parser.add_argument('--metadata-val',
                        default='datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv',
                        type=str, help='path to metadata file (val set)')
    parser.add_argument('--relevancy-path',
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl',
                        type=str, help='path to relevancy matrix (val set)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops in transforms for val')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips for val')
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=2, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
    # Model
    parser.add_argument('--pretrain-model', default='', type=str, help='path to pretrain model')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    parser.add_argument('--find-unused-parameters', action='store_true',
                        help='do this during DDP (useful for models with tied weights)')
    parser.add_argument('--drop-path-rate', default=0.1, type=float, help='drop path ratio')
    parser.add_argument('--dropout-ratio', default=0.5, type=float, help='dropout ratio for the last linear layer')
    parser.add_argument('--num-classes', default=3806, nargs='+', type=int, help='number of classes for the last linear layer')
    parser.add_argument('--use-vn-classifier', action='store_true')
    parser.add_argument('--use-half', action='store_true', help='use half precision at inference')
    # Training
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=16, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--use-sgd', action='store_true')
    parser.add_argument('--freeze-temperature', action='store_true', help='freeze temperature if set to True')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--fix-lr', action='store_true', help='disable cosine lr decay if set True')
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--lr-multiplier-on-backbone', default=0.1, type=float, help='lr multiplier for the backbone')
    parser.add_argument('--clip-grad-type', default='norm', choices=['norm', 'value'])
    parser.add_argument('--clip-grad-value', default=None, type=float, help='')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.01, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--label-smoothing', default=0.1, type=float, help='label smoothing')
    parser.add_argument('--eval-freq', default=5, type=int)
    parser.add_argument('--save-freq', default=5, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--use-zero', action='store_true',
                        help='use ZeroRedundancyOptimizer to save memory')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help='use gradient checkpointing during training for significantly less GPU usage')
    # System
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    return parser


def main(args):
    dist_utils.init_distributed_mode(args)

    global best_acc1
    random_seed(args.seed, dist_utils.get_rank())

    if args.pretrain_model:
        ckpt_path = args.pretrain_model
    else:
        raise Exception('no checkpoint found')
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if args.use_vn_classifier:
        assert args.dataset == 'ek100_cls' and len(args.num_classes) == 3

    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = getattr(models, old_args.model)(
        pretrained=old_args.load_visual_pretrained,
        pretrained2d=old_args.load_visual_pretrained is not None,
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        timesformer_gated_xattn=False,
        timesformer_freeze_space=False,
        num_frames=args.clip_length,
        drop_path_rate=args.drop_path_rate,
    )
    if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
        # inflate weight
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=args.clip_length,
            load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))

    if args.use_vn_classifier:
        model = models.VideoClassifierMultiHead(
            model.visual,
            dropout=args.dropout_ratio,
            num_classes_list=args.num_classes
        )
    else:
        assert len(args.num_classes) == 1
        model = models.VideoClassifier(
            model.visual,
            dropout=args.dropout_ratio,
            num_classes=args.num_classes[0]
        )

    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=args.find_unused_parameters
        )

    p_wd, p_non_wd = [], []
    p_head_wd, p_head_non_wd = [], []
    for n, p in model.named_parameters():
        if 'fc_cls' in n:
            if 'bias' in n:
                p_head_non_wd.append(p)
            else:
                p_head_wd.append(p)
        elif not p.requires_grad:
            continue  # frozen weights
        elif p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [
        {"params": p_wd, "weight_decay": args.wd,  "lr": args.lr * args.lr_multiplier_on_backbone},
        {"params": p_non_wd, "weight_decay": 0, "lr": args.lr * args.lr_multiplier_on_backbone},
        {"params": p_head_wd, "weight_decay": args.wd},
        {"params": p_head_non_wd, "weight_decay": 0}
    ]

    if args.use_zero:
        optimizer = ZeroRedundancyOptimizer(
            optim_params, optimizer_class=torch.optim.SGD if args.use_sgd else torch.optim.AdamW,
            lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd
        )
    else:
        if args.use_sgd:
            optimizer = torch.optim.SGD(optim_params, lr=args.lr, momentum=args.betas[0], weight_decay=args.wd)
        else:
            optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                          eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)
    # optionally resume from a checkpoint (takes precedence over autoresume)
    latest = os.path.join(args.output_dir, 'checkpoint.pt')
    if os.path.isfile(latest):
        args.resume = ''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            if not args.distributed:
                state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k.replace('module.', '')] = v
                result = model.load_state_dict(state_dict, strict=False)
            else:
                result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1']
            print("=> loaded resume checkpoint '{}' (epoch {}, best_metric = {})"
                  .format(args.resume, epoch, best_acc1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")
    if old_args.model.endswith('DISTILBERT_BASE'):
        tokenizer = MyDistilBertTokenizer('distilbert-base-uncased')
    elif old_args.model.endswith('BERT_BASE'):
        tokenizer = MyBertTokenizer('bert-base-uncased')
    elif old_args.model.endswith('BERT_LARGE'):
        tokenizer = MyBertTokenizer('bert-large-uncased')
    elif old_args.model.endswith('GPT2'):
        tokenizer = MyGPT2Tokenizer('gpt2')
    elif old_args.model.endswith('GPT2_MEDIUM'):
        tokenizer = MyGPT2Tokenizer('gpt2-medium')
    elif old_args.model.endswith('GPT2_LARGE'):
        tokenizer = MyGPT2Tokenizer('gpt2-large')
    elif old_args.model.endswith('GPT2_XL'):
        tokenizer = MyGPT2Tokenizer('gpt2-xl')
    else:
        print("Using SimpleTokenizer because of model '{}'. "
              "Please check if this is what you want".format(old_args.model))
        tokenizer = SimpleTokenizer()

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

    crop_size = 224 if '336PX' not in old_args.model else 336
    transforms_list = [
        Permute([3, 0, 1, 2]),    # T H W C -> C T H W
        transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if 'OPENAI' in old_args.model:
        transforms_list.append(transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]))
    else:
        transforms_list.append(transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
    train_transform = transforms.Compose(transforms_list)

    val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in old_args.model else
             transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
            TemporalCrop(frames_per_clip=args.clip_length, stride=args.clip_length),
            SpatialCrop(crop_size=crop_size, num_crops=args.num_crops),
        ])

    # build dataset
    _, mapping_vn2act = generate_label_map(args.dataset)
    if args.dataset == 'ek100_cls':
        args.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
        args.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}
        args.actions = pd.DataFrame.from_dict({'verb': args.mapping_act2v.values(), 'noun': args.mapping_act2n.values()})
    num_clips_at_val = args.num_clips
    args.num_clips = 1
    train_dataset = datasets.get_downstream_dataset(
        train_transform, tokenizer, args, subset='train', label_mapping=mapping_vn2act,
    )
    args.num_clips = num_clips_at_val
    val_dataset = datasets.get_downstream_dataset(
        val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)  # disable distributed
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True
    )
    print('len(train_loader) = {}'.format(len(train_loader)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )
    print('len(val_loader) = {}'.format(len(val_loader)))

    if args.evaluate:
        if args.use_vn_classifier:
            val_stats = validate_multihead(val_loader, model, args)
        else:
            val_stats = validate(val_loader, model, args)
        return

    if args.fix_lr:
        lr_schedule = None
    else:
        lr_schedule = cosine_scheduler(
            args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
            warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start,
        )

    if dist_utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='LaViLa', id=wandb_id, config=args, resume='allow')

    print(args)

    best_metric = 0.
    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)

        is_epoch = ((epoch + 1) % args.save_freq) == 0

        print('=> saving checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'best_acc1': 0,
            'args': args,
        }, False, args.output_dir, is_epoch=is_epoch)

        if ((epoch + 1) % args.eval_freq) == 0:
            if args.use_vn_classifier:
                val_stats = validate_multihead(val_loader, model, args)
            else:
                val_stats = validate(val_loader, model, args)
            if val_stats['acc1'] > best_metric:
                is_best = True
                best_metric = val_stats['acc1']
            else:
                is_best = False

            print('=> saving checkpoint')
            dist_utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc1': best_metric,
                'args': args,
            }, is_best, args.output_dir, is_epoch=is_epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch}

            if dist_utils.is_main_process():
                if args.wandb:
                    wandb.log(log_stats)
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    iters_per_epoch = len(train_loader) // args.update_freq
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_noun = AverageMeter('Noun Acc@1', ':6.2f')
    top1_verb = AverageMeter('Verb Acc@1', ':6.2f')
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, losses, top1, top5, top1_noun, top1_verb],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, (images, target) in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it] * args.lr_multiplier_on_backbone
            else:
                param_group['lr'] = lr_schedule[it]

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            output = model(images, use_checkpoint=args.use_checkpoint)
            if isinstance(output, list):
                assert len(output) == 3
                target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                loss = criterion(output[0], target_to_verb)
                target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                loss += criterion(output[1], target_to_noun)
                loss += criterion(output[2], target)
            else:
                loss = criterion(output, target)
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        if args.clip_grad_value is not None:
            scaler.unscale_(optimizer)
            if args.clip_grad_type == 'norm':
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad_value, norm_type=2.
                )
            elif args.clip_grad_type == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
            else:
                assert False, f"Unknown clip mode ({args.clip_grad_type})."
        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        if isinstance(output, list):
            target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
            acc1_verb, _ = accuracy(output[0], target_to_verb, topk=(1, 5))
            top1_verb.update(acc1_verb.item(), images.size(0))
            target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
            acc1_noun, _ = accuracy(output[1], target_to_noun, topk=(1, 5))
            top1_noun.update(acc1_noun.item(), images.size(0))
            acc1, acc5 = accuracy(output[2], target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
        else:
            output = torch.softmax(output, dim=1)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            if args.dataset == 'ek100_cls':
                vi = get_marginal_indexes(args.actions, 'verb')
                ni = get_marginal_indexes(args.actions, 'noun')
                verb_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), vi)).cuda(args.gpu, non_blocking=True)
                noun_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), ni)).cuda(args.gpu, non_blocking=True)
                target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
                acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
                acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
                top1_verb.update(acc1_verb.item(), images.size(0))
                top1_noun.update(acc1_noun.item(), images.size(0))
            else:
                top1_verb.update(0., images.size(0))
                top1_noun.update(0., images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if dist_utils.is_main_process() and args.wandb:
                wandb.log({
                    'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
                    'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
                })
            progress.display(optim_iter)
    progress.synchronize()
    return {
        'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg,
        'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg,
        'lr': optimizer.param_groups[0]['lr'],
    }


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: '
    )

    # switch to eval mode
    model.eval()
    if args.use_half:
        model.half()

    all_outputs = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if isinstance(images, list):
                logit_allcrops = []
                for crop in images:
                    crop = crop.cuda(args.gpu, non_blocking=True)
                    if args.use_half:
                        crop = crop.half()
                    logit = model(crop, use_checkpoint=args.use_checkpoint)
                    logit_allcrops.append(logit)
                logit_allcrops = torch.stack(logit_allcrops, 0)
                logit = logit_allcrops.mean(0)
                logit = torch.softmax(logit, dim=1)
                target = target.cuda(args.gpu, non_blocking=True)

                acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                top1.update(acc1.item(), target.size(0))
                top5.update(acc5.item(), target.size(0))
            else:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
                if args.use_half:
                    images = images.half()

                logit = model(images, use_checkpoint=args.use_checkpoint)
                logit = torch.softmax(logit, dim=1)

                acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

            all_outputs.append(logit)
            all_targets.append(target)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    progress.synchronize()
    if args.dataset == 'ek100_cls':
        print('EK100 * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    else:
        print('EGTEA * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    all_outputs = torch.cat(all_outputs).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    cm = confusion_matrix(all_targets, all_outputs.argmax(axis=1))
    mean_acc, acc = get_mean_accuracy(cm)
    print('Mean Acc. = {:.3f}, Top-1 Acc. = {:.3f}'.format(mean_acc, acc))

    if args.dataset == 'ek100_cls':
        vi = get_marginal_indexes(args.actions, 'verb')
        ni = get_marginal_indexes(args.actions, 'noun')
        verb_scores = marginalize(all_outputs, vi)
        noun_scores = marginalize(all_outputs, ni)
        target_to_verb = np.array([args.mapping_act2v[a] for a in all_targets.tolist()])
        target_to_noun = np.array([args.mapping_act2n[a] for a in all_targets.tolist()])
        cm = confusion_matrix(target_to_verb, verb_scores.argmax(axis=1))
        _, acc = get_mean_accuracy(cm)
        print('Verb Acc@1: {:.3f}'.format(acc))
        cm = confusion_matrix(target_to_noun, noun_scores.argmax(axis=1))
        _, acc = get_mean_accuracy(cm)
        print('Noun Acc@1: {:.3f}'.format(acc))
    return {'acc1': top1.avg, 'acc5': top5.avg, 'mean_acc': mean_acc}


def validate_multihead(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_verb = AverageMeter('Verb Acc@1', ':6.2f')
    top1_noun = AverageMeter('Noun Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5, top1_verb, top1_noun],
        prefix='Test: '
    )

    # switch to eval mode
    model.eval()
    if args.use_half:
        model.half()

    all_verb_outputs = []
    all_noun_outputs = []
    all_action_outputs = []
    all_verb_targets = []
    all_noun_targets = []
    all_action_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if isinstance(images, torch.Tensor):
                images = [images, ]
            logit_verb_allcrops = []
            logit_noun_allcrops = []
            logit_action_allcrops = []
            for crop in images:
                crop = crop.cuda(args.gpu, non_blocking=True)
                if args.use_half:
                    crop = crop.half()
                logit = model(crop, use_checkpoint=args.use_checkpoint)
                logit_verb_allcrops.append(logit[0])
                logit_noun_allcrops.append(logit[1])
                logit_action_allcrops.append(logit[2])
            logit_verb_allcrops = torch.stack(logit_verb_allcrops, 0)
            logit_noun_allcrops = torch.stack(logit_noun_allcrops, 0)
            logit_action_allcrops = torch.stack(logit_action_allcrops, 0)
            logit_verb = logit_verb_allcrops.mean(0)
            logit_noun = logit_noun_allcrops.mean(0)
            logit_action = logit_action_allcrops.mean(0)
            logit_noun = torch.softmax(logit_noun, dim=1)
            logit_verb = torch.softmax(logit_verb, dim=1)
            logit_action = torch.softmax(logit_action, dim=1)
            target = target.cuda(args.gpu, non_blocking=True)
            target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
            target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)

            acc1, acc5 = accuracy(logit_action, target, topk=(1, 5))
            acc1_verb, _ = accuracy(logit_verb, target_to_verb, topk=(1, 5))
            acc1_noun, _ = accuracy(logit_noun, target_to_noun, topk=(1, 5))
            top1.update(acc1.item(), target.size(0))
            top5.update(acc5.item(), target.size(0))
            top1_verb.update(acc1_verb.item(), target_to_verb.size(0))
            top1_noun.update(acc1_noun.item(), target_to_noun.size(0))

            all_verb_outputs.append(logit_verb)
            all_noun_outputs.append(logit_noun)
            all_action_outputs.append(logit_action)
            all_verb_targets.append(target_to_verb)
            all_noun_targets.append(target_to_noun)
            all_action_targets.append(target)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    progress.synchronize()
    print('EK100 * Verb Acc@1 {top1.avg:.3f}'.format(top1=top1_verb))
    print('EK100 * Noun Acc@1 {top1.avg:.3f}'.format(top1=top1_noun))
    print('EK100 * Action Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return {'acc1': top1.avg, 'acc5': top5.avg, 'acc1_verb': top1_verb.avg, 'acc1_noun': top1_noun.avg}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila finetune and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
