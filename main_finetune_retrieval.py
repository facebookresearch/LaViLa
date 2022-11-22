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
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import wandb

from lavila.data import datasets
from lavila.data.video_transforms import Permute
from lavila.models import models, loss
from lavila.models.tokenizer import (MyBertTokenizer, MyDistilBertTokenizer, MyGPT2Tokenizer, SimpleTokenizer)
from lavila.models.utils import inflate_positional_embeds
from lavila.utils import distributed as dist_utils
from lavila.utils.evaluation_charades import charades_map
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils.preprocess import generate_label_map
from lavila.utils.random import random_seed
from lavila.utils.scheduler import cosine_scheduler
from lavila.utils.evaluation_ek100mir import (calculate_k_counts, calculate_IDCG, calculate_mAP, calculate_nDCG)


def get_args_parser():
    parser = argparse.ArgumentParser(description='lavila finetune and evaluation', add_help=False)
    # Data
    parser.add_argument('--dataset', default='ek100_mir', type=str,
                        choices=['ek100_mir', 'charades_ego'])
    parser.add_argument('--root',
                        default='datasets/EK100/video_ht256px/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata',
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv',
                        type=str, help='path to metadata file (train set)')
    parser.add_argument('--metadata-val',
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv',
                        type=str, help='path to metadata file (val set)')
    parser.add_argument('--relevancy-path',
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl',
                        type=str, help='path to relevancy matrix (val set)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=4, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
    # Model
    parser.add_argument('--pretrain-model', default='', type=str, help='path to pretrain model')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    parser.add_argument('--find-unused-parameters', action='store_true',
                        help='do this during DDP (useful for models with tied weights)')
    parser.add_argument('--drop-path-rate', default=0.1, type=float, help='drop path ratio')
    # Training
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=16, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--freeze-temperature', action='store_true', help='freeze temperature if set to True')
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--fix-lr', action='store_true', help='disable cosine lr decay if set True')
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--clip-grad-type', default='norm', choices=['norm', 'value'])
    parser.add_argument('--clip-grad-value', default=None, type=float, help='')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.01, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
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
    model.logit_scale.requires_grad = False
    model.cuda(args.gpu)
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

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=args.find_unused_parameters
        )

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    if args.use_zero:
        optimizer = ZeroRedundancyOptimizer(
            optim_params, optimizer_class=torch.optim.AdamW,
            lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd
        )
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
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
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

    if args.dataset == 'ek100_mir':
        criterion = loss.MaxMarginRankingLoss(margin=0.2, fix_norm=True).cuda(args.gpu)
    elif args.dataset == 'charades_ego':
        criterion = loss.CLIPLoss(
            use_vissl=True,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size
        )

    crop_size = 224 if '336PX' not in old_args.model else 336
    transforms_list = [
        Permute([3, 0, 1, 2]),    # T H W C -> C T H W
        transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
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
        ])

    # build dataset
    args.model = old_args.model
    args.norm_embed = old_args.norm_embed
    if args.dataset == 'ek100_mir':
        train_dataset = datasets.get_dataset(train_transform, tokenizer, args, is_training=True)
        args.metadata = args.metadata.replace('train', 'test')
        val_dataset = datasets.get_dataset(val_transform, tokenizer, args, is_training=False)
        args.metadata = args.metadata.replace('test', 'train')
    elif args.dataset == 'charades_ego':
        train_dataset = datasets.VideoCaptionDatasetCLIP(
            'charades_ego_trimmed', args.root, args.metadata,
            transform=train_transform, is_training=True, tokenizer=tokenizer,
            clip_length=args.clip_length, clip_stride=args.clip_stride
        )
        labels, mapping_vn2act = generate_label_map(args.dataset)
        val_dataset = datasets.VideoClassyDataset(
            args.dataset, args.root, args.metadata_val,
            transform=val_transform, is_training=False,
            label_mapping=mapping_vn2act, is_trimmed=False,
            num_clips=1, clip_length=args.clip_length, clip_stride=args.clip_stride,
            sparse_sample=args.sparse_sample,
        )
    else:
        raise NotImplementedError

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
        if args.dataset == 'ek100_mir':
            _ = validate_mir(val_loader, model, criterion, args)
        elif args.dataset == 'charades_ego':
            _ = validate_cls(val_loader, ['{}'], labels, model, tokenizer, args)
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

    print("=> zero-shot testing")
    if args.dataset == 'ek100_mir':
        _ = validate_mir(val_loader, model, criterion, args)
    elif args.dataset == 'charades_ego':
        _ = validate_cls(val_loader, ['{}'], labels, model, tokenizer, args)

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

        if (epoch + 1) % args.eval_freq != 0:
            continue

        # TODO: add evaluation
        if args.dataset == 'ek100_mir':
            val_stats = validate_mir(val_loader, model, criterion, args)
        elif args.dataset == 'charades_ego':
            val_stats = validate_cls(val_loader, ['{}'], labels, model, tokenizer, args)

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
    if args.dataset == 'ek100_mir':
        metric_names = ['loss', 'max_margin_loss']
    elif args.dataset == 'charades_ego':
        metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]

        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
        relevancies = inputs.pop()

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(
                *inputs,
                use_checkpoint=args.use_checkpoint,
                norm_embed=args.norm_embed
            )
            if args.dataset == 'ek100_mir':
                loss_dict = criterion(outputs, weight=relevancies)
            elif args.dataset == 'charades_ego':
                loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()
        # TODO: for debug only
        # for n, p in model.named_parameters():
        #     if p.grad is not None:
        #         print('{}: {} | {}'.format(n, torch.mean(torch.abs(p.data)), torch.mean(torch.abs(p.grad))), flush=True)
        #     else:
        #         print('{}: {} | {}'.format(n, torch.mean(torch.abs(p.data)), 'None'), flush=True)
        # if torch.isnan(loss):
        #     for n, p in model.named_parameters():
        #         print(f'{n}:', p.grad, flush=True)

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

        if hasattr(dist_utils.get_model(model), 'logit_scale'):
            # clamp logit scale to [0, 100]
            dist_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = dist_utils.get_model(model).logit_scale.exp().item()
        else:
            logit_scale = torch.nan

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if dist_utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                           'scaler': scaler.get_scale(), 'logit': logit_scale})
            progress.display(optim_iter)
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def validate_mir(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'max_margin_loss']
    iters_per_epoch = len(val_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Test: "
    )

    # switch to eval mode
    model.eval()

    all_video_embed = []
    all_text_embed = []
    with torch.no_grad():
        end = time.time()
        for i, inputs in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
            relevancies = inputs.pop()

            # compute output
            outputs = model(
                *inputs,
                use_checkpoint=args.use_checkpoint,
                norm_embed=args.norm_embed
            )
            loss_dict = criterion(outputs, weight=relevancies)

            for k in loss_dict:
                metrics[k].update(loss_dict[k].item(), args.batch_size)

            image_features = outputs['image_embed']
            text_features = outputs['text_embed']
            all_video_embed.append(image_features.cpu().numpy())
            all_text_embed.append(text_features.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if i % args.print_freq == 0:
                if dist_utils.is_main_process() and args.wandb:
                    wandb.log({**{k: v.item() for k, v in loss_dict.items()}})
                progress.display(i)
    progress.synchronize()
    all_text_embed = np.vstack(all_text_embed)
    all_video_embed = np.vstack(all_video_embed)
    similarity_matrix = np.matmul(all_video_embed, all_text_embed.T)
    similarity_matrix = (similarity_matrix + 1) / 2
    video_id = pd.read_csv(args.metadata.replace('train', 'test')).values[:, 0]
    text_id = pd.read_csv(args.metadata.replace('train', 'test_sentence')).values[:, 0]
    indexes = [video_id.tolist().index(elem) for elem in text_id]
    similarity_matrix = similarity_matrix[:, indexes]
    print(similarity_matrix.shape)
    rel_matrix = pd.read_pickle(
        args.relevancy_path
    )
    vis_map = calculate_mAP(similarity_matrix, rel_matrix)
    txt_map = calculate_mAP(similarity_matrix.T, rel_matrix.T)
    print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, (vis_map + txt_map) / 2))
    vis_k_counts = calculate_k_counts(rel_matrix)
    txt_k_counts = calculate_k_counts(rel_matrix.T)
    vis_IDCG = calculate_IDCG(rel_matrix, vis_k_counts)
    txt_IDCG = calculate_IDCG(rel_matrix.T, txt_k_counts)
    vis_nDCG = calculate_nDCG(similarity_matrix, rel_matrix, k_counts=vis_k_counts, IDCG=vis_IDCG)
    txt_nDCG = calculate_nDCG(similarity_matrix.T, rel_matrix.T, k_counts=txt_k_counts, IDCG=txt_IDCG)
    print('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2))
    return {**{k: v.avg for k, v in metrics.items()}}


def validate_cls(val_loader, templates, labels, model, tokenizer, args):
    # switch to eval mode
    model.eval()

    all_outputs = []
    all_targets = []
    with torch.no_grad():
        text_features = []
        for label in labels:
            if isinstance(label, list):
                texts = [tmpl.format(lbl) for tmpl in templates for lbl in label]
            else:
                texts = [tmpl.format(label) for tmpl in templates]
            texts = tokenizer(texts)
            if isinstance(texts, tuple):
                # Bert-style tokenizer will output both ids and mask
                texts, masks = texts
                texts = texts.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)
            else:
                texts = texts.cuda(non_blocking=True)
                masks = None
            texts = texts.view(-1, 77).contiguous()
            masks = masks.view(-1, 77).contiguous() if masks is not None else None
            if masks is not None:
                class_embeddings = dist_utils.get_model(model).encode_text(texts, attention_mask=masks)
            else:
                class_embeddings = dist_utils.get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        print('=> start forwarding')
        end_time = time.time()
        for i, (images, target) in enumerate(val_loader):
            if i % args.print_freq == 0:
                print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                end_time = time.time()
            if isinstance(images, torch.Tensor):
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # encode images
                image_features = dist_utils.get_model(model).encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # cosine similarity as logits
                logits_per_image = image_features @ text_features.t()
                logits_per_image = torch.softmax(logits_per_image, dim=1)
            else:
                target = target.cuda(non_blocking=True)
                images_list = images
                logits_all_clips = []
                for images in images_list:
                    images = images.cuda(non_blocking=True)
                    image_features = dist_utils.get_model(model).encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logits_per_image = image_features @ text_features.t()
                    logits_all_clips.append(logits_per_image)

                logits_all_clips = torch.stack(logits_all_clips, dim=0)
                # logits_per_image = logits_all_clips.max(0).values
                logits_per_image = logits_all_clips.mean(0)
                logits_per_image = torch.softmax(logits_per_image, dim=1)

            all_outputs.append(logits_per_image.cpu())
            all_targets.append(target.cpu())
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    preds, targets = all_outputs.numpy(), all_targets.numpy()
    m_ap, _, _ = charades_map(preds, targets)
    print('mAP = {:.3f}'.format(m_ap))
    return {'mAP': m_ap}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila finetune and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
