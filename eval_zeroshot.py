# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import os.path as osp
import time
from collections import OrderedDict

import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from sklearn.metrics import confusion_matrix

from lavila.data import datasets
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from lavila.models import models
from lavila.models.utils import inflate_positional_embeds
from lavila.utils import distributed as dist_utils
from lavila.utils.evaluation import accuracy, get_mean_accuracy
from lavila.utils.evaluation_egomcq import egomcq_accuracy_metrics
from lavila.utils.evaluation_ek100mir import (calculate_k_counts, calculate_IDCG, calculate_mAP, calculate_nDCG)
from lavila.utils.evaluation_charades import charades_map
from lavila.utils.preprocess import generate_label_map, generate_tokenizer


def get_args_parser():
    parser = argparse.ArgumentParser(description='LAVILA 0-shot evaluations', add_help=False)
    parser.add_argument('--dataset', default='ek100_mir', type=str,
                        choices=['ek100_cls', 'ek100_mir', 'charades_ego', 'egtea', 'ego4d_mcq'])
    parser.add_argument('--root',
                        default='datasets/EK100/video_ht256px/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata-val',
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv',
                        type=str, help='path to metadata file (val set)')
    parser.add_argument('--relevancy-path',
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl',
                        type=str, help='path to relevancy matrix (val set)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops in transforms')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips (for untrimmed videos, eg. Charades)')
    parser.add_argument('--clip-length', default=4, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=16, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
    parser.add_argument('--batch-size', default=16, type=int, help='batch_size')
    parser.add_argument('--cls-use-template', action='store_true', help='use prompt in 0-shot classification')
    parser.add_argument('--print-freq', default=100, type=int)
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--use-half', action='store_true')
    return parser


def main(args):
    if args.resume:
        ckpt_path = args.resume
    elif osp.isfile(osp.join(args.output_dir, 'checkpoint_best.pt')):
        ckpt_path = osp.join(args.output_dir, 'checkpoint_best.pt')
    else:
        raise Exception('no checkpoint found')

    ckpt = torch.load(ckpt_path, map_location='cpu')

    # create model
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    print('=> creating model: {}'.format(old_args.model))
    model = getattr(models, old_args.model)(
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        gated_xattn=False if 'gated_xattn' not in old_args else old_args.gated_xattn,
        timesformer_gated_xattn=False if 'timesformer_gated_xattn' not in old_args else old_args.timesformer_gated_xattn,
        timesformer_freeze_space=False if 'timesformer_freeze_space' not in old_args else old_args.timesformer_freeze_space,
        freeze_lm_vclm=False if 'freeze_lm_vclm' not in old_args else old_args.freeze_lm_vclm,
        freeze_visual_vclm=False if 'freeze_visual_vclm' not in old_args else old_args.freeze_visual_vclm,
        num_frames=args.clip_length,
        drop_path_rate=0,
    )
    model.cuda()
    if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
        # inflate weight
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=args.clip_length,
            load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {}, best_metric = {})".format(args.resume, ckpt['epoch'], ckpt['best_acc1']))

    torch.backends.cudnn.benchmark = True

    if args.dataset in ['ek100_cls', 'charades_ego', 'egtea']:
        labels, mapping_vn2act = generate_label_map(args.dataset)
    else:
        mapping_vn2act = None
    tokenizer = generate_tokenizer(old_args.model)
    crop_size = 224 if '336PX' not in old_args.model else 336
    if args.num_crops == 1 and args.num_clips == 1:
        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if ('OPENAI' not in old_args.model) else
             transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
        ])
    else:
        val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.Resize(crop_size),
            (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if ('OPENAI' not in old_args.model) else
             transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
            TemporalCrop(frames_per_clip=args.clip_length, stride=args.clip_length),
            SpatialCrop(crop_size=crop_size, num_crops=args.num_crops),
            ])

    val_dataset = datasets.get_downstream_dataset(
        val_transform, tokenizer, args, subset='val', label_mapping=mapping_vn2act,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    if args.cls_use_template:
        templates = ['#C C {}', '#C {}']
    else:
        templates = ['{}']

    if args.dataset in ['ek100_cls', 'charades_ego', 'egtea']:
        preds, targets = validate_zeroshot(val_loader, templates, labels, model, tokenizer)
    if args.dataset == 'ek100_cls':
        if args.use_half:
            preds = preds.float()
        top1, top5 = accuracy(preds, targets, topk=(1, 5))
        print('top1 = {:.3f}'.format(top1.item()))
        print('top5 = {:.3f}'.format(top5.item()))
    elif args.dataset == 'charades_ego':
        preds, targets = preds.numpy(), targets.numpy()
        m_ap, _, _ = charades_map(preds, targets)
        print('mAP = {:.3f}'.format(m_ap))
    elif args.dataset == 'egtea':
        preds, targets = preds.numpy(), targets.numpy()
        print(preds.shape, targets.shape)
        cm = confusion_matrix(targets, preds.argmax(axis=1))
        mean_class_acc, acc = get_mean_accuracy(cm)
        print('Mean Acc. = {:.3f}, Top-1 Acc. = {:.3f}'.format(mean_class_acc, acc))

    if args.dataset == 'ek100_mir':
        val_dataset = datasets.VideoCaptionDatasetCLIP(
            'ek100_mir',
            args.root,
            args.metadata_val,
            transform=val_transform, is_training=False,
            tokenizer=tokenizer,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
            sparse_sample=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False
        )
        similarity_matrix = get_similarity_matrix(val_loader, model, print_freq=args.print_freq, use_half=args.use_half)
        similarity_matrix = (similarity_matrix + 1) / 2
        video_id = pd.read_csv(args.metadata_val).values[:, 0]
        text_id = pd.read_csv(args.metadata_val.replace("test.csv", "test_sentence.csv")).values[:, 0]
        indexes = [video_id.tolist().index(elem) for elem in text_id]
        similarity_matrix = similarity_matrix[:, indexes]
        print(similarity_matrix.shape)
        rel_matrix = pd.read_pickle(args.relevancy_path)
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

    if args.dataset == 'ego4d_mcq':
        val_dataset = datasets.VideoCaptionDatasetMCQ(
            args.dataset,
            args.root,
            args.metadata_val,
            transform=val_transform, is_training=False,
            tokenizer=tokenizer,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
            sparse_sample=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False
        )
        validate_mcq(val_loader, model, use_half=args.use_half)


def validate_zeroshot(val_loader, templates, labels, model, tokenizer):
    model.eval()
    if args.use_half:
        model = model.half()
    all_outputs = []
    all_targets = []
    all_vis_features = []
    print('=> encoding captions')
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
                if args.use_half:
                    images = images.half()
                target = target.cuda(non_blocking=True)

                # encode images
                image_features = dist_utils.get_model(model).encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_vis_features.append(image_features)
                # cosine similarity as logits
                logits_per_image = image_features @ text_features.t()
                # logits_per_image = torch.softmax(logits_per_image, dim=1)
            else:
                target = target.cuda(non_blocking=True)
                images_list = images
                logits_all_clips = []
                for images in images_list:
                    images = images.cuda(non_blocking=True)
                    if args.use_half:
                        images = images.half()
                    image_features = dist_utils.get_model(model).encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logits_per_image = image_features @ text_features.t()
                    logits_all_clips.append(logits_per_image)

                logits_all_clips = torch.stack(logits_all_clips, dim=0)
                logits_per_image = logits_all_clips.max(0).values
                # logits_per_image = logits_all_clips.mean(0)
                logits_per_image = torch.softmax(logits_per_image, dim=1)

            all_outputs.append(logits_per_image.cpu())
            all_targets.append(target.cpu())

    return torch.cat(all_outputs), torch.cat(all_targets)


def get_similarity_matrix(val_loader, model, print_freq=100, use_half=False):
    model.eval()
    if use_half:
        model = model.half()
    all_text_embed = []
    all_video_embed = []
    with torch.no_grad():
        print('=> encoding visual and textual')
        for i, inputs in enumerate(val_loader):
            if i % print_freq == 0:
                print('finish batch {}/{}'.format(i, len(val_loader)))
            frames = inputs[0].cuda(non_blocking=True)
            if use_half:
                frames = frames.half()
            texts = inputs[1].cuda(non_blocking=True)
            if len(inputs) == 4:
                masks = inputs[2].cuda(non_blocking=True)
            else:
                masks = None

            # encode images
            image_features = dist_utils.get_model(model).encode_image(frames)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_video_embed.append(image_features.cpu().numpy())

            if texts.ndim == 3:
                is_multiple_narrations = True
                texts = texts.view(-1, texts.shape[-1])
            else:
                is_multiple_narrations = False
            if masks is not None:
                text_features = dist_utils.get_model(model).encode_text(texts, attention_mask=masks)
            else:
                text_features = dist_utils.get_model(model).encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_text_embed.append(text_features.cpu().numpy())

        all_text_embed = np.vstack(all_text_embed)
        all_video_embed = np.vstack(all_video_embed)
        similarity_matrix = np.matmul(all_video_embed, all_text_embed.T)
        if is_multiple_narrations:
            similarity_matrix = similarity_matrix.reshape(all_video_embed.shape[0], all_video_embed.shape[0], -1)

    return similarity_matrix


def validate_mcq(val_loader, model, use_half=False):
    model.eval()
    if use_half:
        model.half()
    with torch.no_grad():
        print('=> start forwarding')
        all_preds = []
        all_gts = []
        all_types = []
        end_time = time.time()
        for i, inputs in enumerate(val_loader):
            if i % args.print_freq == 0:
                print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                end_time = time.time()
            texts_query = inputs[0].cuda(non_blocking=True)
            frames_options = inputs[1].cuda(non_blocking=True)
            if use_half:
                frames_options = frames_options.half()
            answer = inputs[3]
            q_type = inputs[4]
            if len(inputs) == 7:
                masks_query = inputs[5].cuda(non_blocking=True)
            else:
                masks_query = None

            batch_size = frames_options.shape[0]

            frames_options = frames_options.view(-1, *frames_options.shape[2:])
            image_features = dist_utils.get_model(model).encode_image(frames_options)
            image_features = image_features.view(batch_size, -1, *image_features.shape[1:])

            if masks_query is not None:
                query_features = dist_utils.get_model(model).encode_text(texts_query, attention_mask=masks_query)
            else:
                query_features = dist_utils.get_model(model).encode_text(texts_query)

            all_gts.append(answer)
            all_types.append(q_type)
            for j in range(batch_size):
                similarity_matrix = torch.matmul(query_features[j], image_features[j].T)
                similarity_matrix = similarity_matrix.cpu().detach()
                all_preds.append(similarity_matrix)
        all_preds = torch.stack(all_preds)
        all_gts = torch.cat(all_gts)
        all_types = torch.cat(all_types)
        metrics = egomcq_accuracy_metrics(all_preds, all_gts, all_types)
        print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila 0-shot evaluations', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
