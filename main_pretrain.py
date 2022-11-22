# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import math
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

from eval_zeroshot import get_similarity_matrix
from lavila.data import datasets
from lavila.data.video_transforms import Permute
from lavila.models import models
from lavila.utils.meter import AverageMeter, ProgressMeter
from lavila.utils import distributed as dist_utils
from lavila.utils.evaluation_ek100mir import get_mAP, get_nDCG
from lavila.utils.preprocess import generate_tokenizer
from lavila.utils.random import random_seed
from lavila.utils.scheduler import cosine_scheduler


class GroundTruthDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return 1, self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return 0, self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def get_args_parser():
    parser = argparse.ArgumentParser(description='LaVid training and evaluation', add_help=False)
    # Data
    parser.add_argument('--dataset', default='ego4d', type=str, choices=['ego4d'])
    parser.add_argument('--root', default='datasets/Ego4D/video_5min_chunks_288px/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata', default='datasets/Ego4D/ego4d_train.pkl',
                        type=str, help='path to metadata file')
    parser.add_argument('--metadata-aux', default=None, nargs='+',
                        type=str, help='path to metadata file (auxiliary data with pseudo narrations)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--clip-length', default=4, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=16, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
    parser.add_argument('--narration-selection', default='random',
                        choices=['random', 'concat'],
                        type=str, help='selection strategy if multiple narrations per clip')
    parser.add_argument('--num-hard-neg', default=0, type=int, help='number of hard negatives per video')
    # Model
    parser.add_argument('--model', default='CLIP_OPENAI_TIMESFORMER_BASE', type=str)
    parser.add_argument('--norm-embed', action='store_true', help='norm text and visual embed if set True')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    parser.add_argument('--load-visual-pretrained', default=None, type=str,
                        help='path to pretrained model (in1k/in21k/...)')
    parser.add_argument('--project-embed-dim', default=256, type=int, help='embed dim after projection')
    parser.add_argument('--use-cls-token', action='store_true', help='use feature at [CLS] if set True')
    parser.add_argument('--contrastive-use-vissl', action='store_true', help='use contrastive implementation in vissl')
    parser.add_argument('--gated-xattn', action='store_true', help='use gated x-attn in VCLM_GPT2')
    parser.add_argument('--random-init-gpt2', action='store_true', help='random initialize params of text decoder in VCLM_GPT2')
    parser.add_argument('--timesformer-gated-xattn', action='store_true', help='use gated x-attn in TimeSformer')
    parser.add_argument('--timesformer-freeze-space', action='store_true', help='freeze space part in TimeSformer')
    parser.add_argument('--drop-path-rate', default=0., type=float, help='DropPath rate')
    parser.add_argument('--freeze-visual-vclm', action='store_true', help='freeze the visual model in VCLM_GPT2')
    parser.add_argument('--freeze-visual-vclm-temporal', action='store_true', help='freeze the temporal part of visual model in VCLM_GPT2')
    parser.add_argument('--freeze-lm-vclm', action='store_true', help='freeze the lm in VCLM_GPT2')
    parser.add_argument('--find-unused-parameters', action='store_true',
                        help='do this during DDP (useful for models with tied weights)')
    # Training
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=32, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--temperature-init', default=0.07, type=float,
                        help='init. logit temperature for samples')
    parser.add_argument('--freeze-temperature', action='store_true',
                        help='freeze logit temperature')
    parser.add_argument('--pseudo-temperature-init', default=0.07, type=float,
                        help='init. logit temperature for pseudo-narrated samples')
    parser.add_argument('--freeze-pseudo-temperature', action='store_true',
                        help='freeze logit temperature (for pseudo-narrated samples)')
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
    parser.add_argument('--eval-freq', default=99, type=int)
    parser.add_argument('--eval-in-middle-freq', default=-1, type=int)
    parser.add_argument('--save-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--use-zero', action='store_true',
                        help='use ZeroRedundancyOptimizer to save memory')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help='use gradient checkpointing during training for significantly less GPU usage')
    parser.add_argument('--use-half', action='store_true', help='evaluate using half-precision')
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
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

    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(
        pretrained=args.load_visual_pretrained,
        pretrained2d=args.load_visual_pretrained is not None,
        text_use_cls_token=args.use_cls_token,
        project_embed_dim=args.project_embed_dim,
        gated_xattn=args.gated_xattn,
        random_init_gpt2=args.random_init_gpt2,
        timesformer_gated_xattn=args.timesformer_gated_xattn,
        timesformer_freeze_space=args.timesformer_freeze_space,
        freeze_lm_vclm=args.freeze_lm_vclm,
        freeze_visual_vclm=args.freeze_visual_vclm,
        freeze_visual_vclm_temporal=args.freeze_visual_vclm_temporal,
        num_frames=args.clip_length,
        drop_path_rate=args.drop_path_rate,
        temperature_init=args.temperature_init,
    )
    if args.freeze_temperature:
        print('Freeze logit temperature')
        model.logit_scale.requires_grad = False
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=args.find_unused_parameters
        )

    tokenizer = generate_tokenizer(args.model)

    if args.metadata_aux is None:
        criterion = models.get_loss(args.model, args, tokenizer=tokenizer).cuda(args.gpu)
    else:
        criterion = models.loss.SSLCLIPLoss(
            use_vissl=args.contrastive_use_vissl,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            scale_init=args.pseudo_temperature_init,
            freeze_scale=args.freeze_pseudo_temperature,
            ).cuda(args.gpu)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
    for n, p in criterion.named_parameters():
        if not p.requires_grad:
            continue
        p_non_wd.append(p)

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
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            criterion.load_state_dict(checkpoint['criterion']) if 'criterion' in checkpoint else ()
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

    crop_size = 224 if '336PX' not in args.model else 336
    transforms_list = [
        Permute([3, 0, 1, 2]),    # T H W C -> C T H W
        transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
    ]
    if 'OPENAI' in args.model:
        transforms_list.append(transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]))
    else:
        transforms_list.append(transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
    train_transform = transforms.Compose(transforms_list)

    # TODO: uncomment when evaluation is done later
    val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in args.model else
             transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]))
        ])

    assert 'train' in args.metadata
    train_dataset = datasets.get_dataset(train_transform, tokenizer, args, is_training=True)
    args.metadata = args.metadata.replace('train', 'val')
    val_dataset = datasets.get_dataset(val_transform, tokenizer, args, is_training=False)
    args.metadata = args.metadata.replace('val', 'train')
    if args.metadata_aux is not None:
        train_dataset = GroundTruthDataset(train_dataset)
        old_metadata = args.metadata
        aux_dataset_list = []
        for aux_i, aux_pkl in enumerate(args.metadata_aux):
            args.metadata = aux_pkl
            aux_dataset = datasets.get_dataset(train_transform, tokenizer, args, is_training=True)
            aux_dataset_list.append(PseudoLabelDataset(aux_dataset))
            print("auxiliary dataset [{}] : source = {}, len(aux_dataset) = {}".format(aux_i, aux_pkl, len(aux_dataset)))
        pseudo_label_dataset = torch.utils.data.ConcatDataset(aux_dataset_list)
        args.metadata = old_metadata
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, pseudo_label_dataset])
        val_dataset = GroundTruthDataset(val_dataset)

    ek100_dataset = datasets.VideoCaptionDatasetCLIP(
        'ek100_mir',
        'datasets/EK100/video_ht256px/',
        'datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv',
        transform=val_transform,
        is_training=False,
        tokenizer=tokenizer,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        sparse_sample=False
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        ek100_sampler = torch.utils.data.SequentialSampler(ek100_dataset)
    else:
        train_sampler = None
        val_sampler = None
        ek100_sampler = None

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
    ek100_loader = torch.utils.data.DataLoader(
        ek100_dataset, batch_size=args.batch_size * (1 + args.num_hard_neg), shuffle=(ek100_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=ek100_sampler, drop_last=False
    )
    print('len(ek100_loader) = {}'.format(len(ek100_loader)))

    if args.fix_lr:
        lr_schedule = None
    else:
        lr_schedule = cosine_scheduler(
            args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
            warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start,
        )

    if dist_utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='LaVid', id=wandb_id, config=args, resume='allow')

    print(args)

    best_metric = 0.
    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if hasattr(args, 'eval_in_middle_freq') and args.eval_in_middle_freq > 0:
            train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args,
                                ek100_loader=ek100_loader, eval_in_middle=args.eval_in_middle_freq)
        else:
            train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)

        if args.model.startswith('CLIP'):
            print('=> 0-shot on EK100')
            similarity_matrix = get_similarity_matrix(ek100_loader, model, use_half=args.use_half)
            similarity_matrix = (similarity_matrix + 1) / 2
            video_id = pd.read_csv("datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv").values[:, 0]
            text_id = pd.read_csv("datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test_sentence.csv").values[:, 0]
            indexes = [video_id.tolist().index(elem) for elem in text_id]
            similarity_matrix = similarity_matrix[:, indexes]
            rel_matrix = pd.read_pickle(
                'datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl'
            )
            vis_map, txt_map, avg_map = get_mAP(similarity_matrix, rel_matrix)
            print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, avg_map))
            vis_ndcg, txt_ndcg, avg_ndcg = get_nDCG(similarity_matrix, rel_matrix)
            print('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_ndcg, txt_ndcg, avg_ndcg))
            if avg_map > best_metric:
                is_best = True
                best_metric = avg_map
            else:
                is_best = False
        else:
            is_best = False

        is_epoch = ((epoch + 1) % args.save_freq) == 0

        if args.distributed and args.use_zero:
            print("=> consolidating state_dict before saving (due to ZeRO)")
            optimizer.consolidate_state_dict()

        print('=> saving checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict() if dist_utils.get_rank() == 0 else {},
            'scaler': scaler.state_dict(),
            'best_acc1': best_metric,
            'args': args,
        }, is_best, args.output_dir, is_epoch=is_epoch)

        if (epoch + 1) % args.eval_freq != 0:
            continue

        # TODO: add evaluation
        val_stats = validate(val_loader, model, criterion, args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        if dist_utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, ek100_loader=None, eval_in_middle=0):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    if args.metadata_aux is not None:
        metric_names.extend(['num_gt', 'num_pseudo', 'clip_acc_gt', 'clip_acc_pseudo'])
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
        # evaluate in the middle of training
        if eval_in_middle > 0 and (data_iter > 0 and data_iter % eval_in_middle) and ek100_loader is not None:
            model.eval()
            print('=> 0-shot on EK100 in the middle of training')
            similarity_matrix = get_similarity_matrix(ek100_loader, model, use_half=args.use_half)
            similarity_matrix = (similarity_matrix + 1) / 2
            video_id = pd.read_csv("datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv").values[:, 0]
            text_id = pd.read_csv("datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test_sentence.csv").values[:, 0]
            indexes = [video_id.tolist().index(elem) for elem in text_id]
            similarity_matrix = similarity_matrix[:, indexes]
            rel_matrix = pd.read_pickle(
                'datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl'
            )
            vis_map, txt_map, avg_map = get_mAP(similarity_matrix, rel_matrix)
            print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, avg_map))
            vis_ndcg, txt_ndcg, avg_ndcg = get_nDCG(similarity_matrix, rel_matrix)
            print('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_ndcg, txt_ndcg, avg_ndcg))
            best_metric = avg_map

            print('=> saving checkpoint')
            dist_utils.save_on_master({
                'epoch': epoch + data_iter / len(train_loader),
                'state_dict': model.state_dict(),
                'criterion': criterion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc1': best_metric,
                'args': args,
            }, False, args.output_dir, is_epoch=True)   # save every time (not to conflict the best_metric tracking in the regular validation phrase)
            model.train()

        if args.metadata_aux is not None:
            gt_indicators, inputs = inputs

        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]

        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
        _ = inputs.pop()  # loader will a "relevancy" variable which is not needed except ek100_mir

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(
                *inputs,
                use_checkpoint=args.use_checkpoint,
                norm_embed=args.norm_embed
            )
            if args.metadata_aux is None:
                loss_dict = criterion(outputs)
            else:
                loss_dict = criterion(outputs, gt_indicators)
            loss = loss_dict['loss']
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


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(val_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Test: "
    )

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, inputs in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.metadata_aux is not None:
                gt_indicators, inputs = inputs

            inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
            _ = inputs.pop()  # loader will a "relevancy" variable which is not needed except ek100_mir

            # compute output
            outputs = model(
                *inputs,
                use_checkpoint=args.use_checkpoint,
                norm_embed=args.norm_embed
            )
            if args.metadata_aux is None:
                loss_dict = criterion(outputs)
            else:
                loss_dict = criterion(outputs, gt_indicators)

            for k in loss_dict:
                metrics[k].update(loss_dict[k].item(), args.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if i % args.print_freq == 0:
                if dist_utils.is_main_process() and args.wandb:
                    wandb.log({**{k: v.item() for k, v in loss_dict.items()}})
                progress.display(i)
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LaVid training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
