# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os.path as osp
import time
from collections import OrderedDict

import numpy as np
# https://github.com/numpy/numpy/issues/21079
try:
    import numpy.distutils
    numpy.distutils.__config__.blas_opt_info = np.distutils.__config__.blas_ilp64_opt_info
except Exception:
    pass
from nlgeval import NLGEval

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from lavila.data import datasets
from lavila.data.video_transforms import Permute, SpatialCrop, TemporalCrop
from lavila.models import models
from lavila.models.utils import inflate_positional_embeds
from lavila.utils import distributed as dist_utils
from lavila.utils.preprocess import generate_tokenizer


def decode_one(generated_ids, tokenizer):
    # get the index of <EOS>
    if tokenizer.eos_token_id == tokenizer.bos_token_id:
        if tokenizer.eos_token_id in generated_ids[1:].tolist():
            eos_id = generated_ids[1:].tolist().index(tokenizer.eos_token_id) + 1
        else:
            eos_id = len(generated_ids.tolist()) - 1
    elif tokenizer.eos_token_id in generated_ids.tolist():
        eos_id = generated_ids.tolist().index(tokenizer.eos_token_id)
    else:
        eos_id = len(generated_ids.tolist()) - 1
    generated_text_str = tokenizer.tokenizer.decode(generated_ids[1:eos_id].tolist())
    return generated_text_str


def get_args_parser():
    parser = argparse.ArgumentParser(description='LAVILA 0-shot evaluations', add_help=False)
    parser.add_argument('--dataset', default='ego4d', type=str,
                        choices=['ego4d'])
    parser.add_argument('--root',
                        default='datasets/Ego4D/video_5min_chunks_288px/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata-val',
                        default='datasets/Ego4D/ego4d_val.pkl',
                        type=str, help='path to metadata file (val set)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops in transforms')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips (for untrimmed videos, eg. Charades)')
    parser.add_argument('--clip-length', default=4, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=16, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
    parser.add_argument('--batch-size', default=16, type=int, help='batch_size')
    # captioning options
    parser.add_argument('--caption-sample', default='multinomial_sample',
                        choices=['multinomial_sample', 'beam_sample', 'group_beam_search'])
    parser.add_argument('--caption-top-k', default=None, type=int, help='top-k sampling (predecessor of nucleus sampling)')
    parser.add_argument('--caption-top-p', default=0.95, type=float, help='top-p sampling sampling (aka nucleus sampling)')
    parser.add_argument('--caption-num-beams', default=3, type=int)
    parser.add_argument('--caption-num-beam-groups', default=1, type=int)
    parser.add_argument('--caption-temperature', default=0.7, type=float)
    parser.add_argument('--caption-length-penalty', default=1.0, type=float)
    parser.add_argument('--caption-num-return-sequences', default=1, type=int)
    parser.add_argument('--caption-max-len', default=77, type=int)
    parser.add_argument('--caption-disable-visual', action='store_true')
    parser.add_argument('--caption-early-stop', action='store_true', help='early stopping to save computation')
    parser.add_argument('--caption-output-filename', default='caption.txt', type=str)
    # others
    parser.add_argument('--eval-freq', default=1000, type=int,
                        help='percentage (1/eval_freq) of val data to evaluate (for fast prototyping)')
    parser.add_argument('--print-freq', default=10, type=int)
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

    val_dataset = datasets.VideoCaptionDatasetCLIP(
            args.dataset,
            args.root,
            args.metadata_val,
            transform=val_transform,
            is_training=False,
            tokenizer=tokenizer,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
            sparse_sample=False,
            subsample_stride=args.eval_freq,
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    validate_caption(val_loader, model, tokenizer, args.caption_output_filename, use_half=args.use_half)


def validate_caption(val_loader, model, tokenizer, output_filename='caption.txt', use_half=False):
    model.eval()
    if args.use_half:
        model = model.half()
    nlgeval = NLGEval()
    f = open(output_filename, 'w')
    ppls_all = []
    ppls_with_teacher_all = []
    reference = []
    hypothesis = []
    end_time = time.time()
    id_offset = 0
    print('=> start forwarding')
    with torch.no_grad():
        for i, inputs in enumerate(val_loader):
            if i % args.print_freq == 0:
                print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                end_time = time.time()
            images = inputs[0].cuda(non_blocking=True)
            if use_half:
                images = images.half()
            target = inputs[1].cuda(non_blocking=True)

            # encode images
            image_features = dist_utils.get_model(model).encode_image(images)

            # teacher forcing (to get standard ppl metric)
            generated_text_ids_with_teacher, ppls_with_teacher = dist_utils.get_model(model).generate(
                image_features,
                tokenizer,
                target=target,
                max_text_length=args.caption_max_len,
                top_k=args.caption_top_k,
                top_p=args.caption_top_p,
                teacher_forcing=True,
                early_stopping=args.caption_early_stop,
            )

            if args.caption_sample == 'multinomial_sample':
                assert args.caption_num_beam_groups == 1
                generated_text_ids, ppls = dist_utils.get_model(model).generate(
                    image_features,
                    tokenizer,
                    target=target.repeat_interleave(args.caption_num_return_sequences, dim=0),
                    max_text_length=args.caption_max_len,
                    top_k=args.caption_top_k,
                    top_p=args.caption_top_p,
                    num_return_sequences=args.caption_num_return_sequences,
                    temperature=args.caption_temperature,
                    early_stopping=args.caption_early_stop,
                )
            elif args.caption_sample == 'beam_sample':
                assert args.caption_num_beam_groups == 1
                generated_text_ids, ppls = dist_utils.get_model(model).beam_sample(
                    image_features,
                    tokenizer,
                    target=target,
                    max_text_length=args.caption_max_len,
                    top_k=args.caption_top_k,
                    top_p=args.caption_top_p,
                    temperature=args.caption_temperature,
                    length_penalty=args.caption_length_penalty,
                    num_beams=args.caption_num_beams,
                    num_return_sequences=args.caption_num_return_sequences,
                    early_stopping=args.caption_early_stop,
                )
            elif args.caption_sample == 'group_beam_search':
                assert args.caption_num_beam_groups > 1 and args.caption_num_beams % args.caption_num_beam_groups == 0
                generated_text_ids, ppls = dist_utils.get_model(model).group_beam_search(
                    image_features,
                    tokenizer,
                    target=target if not args.caption_no_gt else None,
                    max_text_length=args.caption_max_len,
                    top_k=args.caption_top_k,
                    top_p=args.caption_top_p,
                    temperature=args.caption_temperature,
                    length_penalty=args.caption_length_penalty,
                    num_beams=args.caption_num_beams,
                    num_beam_groups=args.caption_num_beam_groups,
                    num_return_sequences=args.caption_num_return_sequences,
                    early_stopping=args.caption_early_stop,
                )
            else:
                raise NotImplementedError
            ppls_all.append(ppls.reshape(-1, args.caption_num_return_sequences).mean(1))
            ppls_with_teacher_all.append(ppls_with_teacher)

            for j in range(generated_text_ids.shape[0] // args.caption_num_return_sequences):
                for k in range(args.caption_num_return_sequences):
                    jj = j * args.caption_num_return_sequences + k

                    generated_text_str = decode_one(generated_text_ids[jj], tokenizer)
                    gt_text = decode_one(target[j], tokenizer)
                    generated_text_str_with_teacher = decode_one(generated_text_ids_with_teacher[j], tokenizer)

                    from transformers import BertTokenizer
                    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    gt_text = bert_tokenizer.decode(bert_tokenizer(gt_text)['input_ids'][1:-1])
                    generated_text_str = bert_tokenizer.decode(bert_tokenizer(generated_text_str)['input_ids'][1:-1])
                    generated_text_str_with_teacher = bert_tokenizer.decode(bert_tokenizer(generated_text_str_with_teacher)['input_ids'][1:-1])
                    reference.append(gt_text)
                    hypothesis.append(generated_text_str)
                    s1 = '[{:6d}] Groundtruth              |                 | {}'.format(id_offset + j, gt_text)
                    s2 = '[{:6d}] Generated                | PPL : {:9.3f} | {}'.format(id_offset + j, ppls[jj], generated_text_str)
                    s3 = '[{:6d}] Generated (w/. teacher)  | PPL : {:9.3f} | {}'.format(id_offset + j, ppls_with_teacher[j], generated_text_str_with_teacher)
                    for s in [s1, s2, s3]:
                        # if i % args.print_freq == 0:
                        #     print(s)
                        f.write('{} \n'.format(s))
            id_offset += generated_text_ids.shape[0] // args.caption_num_return_sequences

    ppls_with_teacher_all = torch.cat(ppls_with_teacher_all, dim=0)
    ppls_all = torch.cat(ppls_all, dim=0)

    print('PPL (w/.  teacher) = {:9.3f}'.format(ppls_with_teacher_all.mean().item()))
    print('PPL (w/o. teacher) = {:9.3f}'.format(ppls_all.mean().item()))
    f.write('PPL (w/.  teacher) = {:9.3f} \n'.format(ppls_with_teacher_all.mean().item()))
    f.write('PPL (w/o. teacher) = {:9.3f} \n'.format(ppls_all.mean().item()))

    print('Avg length for reference:  {:9.3f}'.format(sum(map(lambda sentence: len(sentence.split(' ')), reference)) / len(reference)))
    print('Avg length for hypothesis: {:9.3f}'.format(sum(map(lambda sentence: len(sentence.split(' ')), hypothesis)) / len(hypothesis)))
    f.write('Avg length for reference:  {:9.3f} \n'.format(sum(map(lambda sentence: len(sentence.split(' ')), reference)) / len(reference)))
    f.write('Avg length for hypothesis: {:9.3f} \n'.format(sum(map(lambda sentence: len(sentence.split(' ')), hypothesis)) / len(hypothesis)))

    print('=> Calling NLGEval')
    f.write('=> Calling NLGEval\n')
    metrics_dict = nlgeval.compute_metrics([reference], hypothesis)
    for k in metrics_dict:
        print('{:16s} = {:9.3f}'.format(k, metrics_dict[k]))
        f.write('{:16s} = {:9.3f} \n'.format(k, metrics_dict[k]))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila 0-shot evaluations', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
