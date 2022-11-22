# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from collections import OrderedDict
import os
import os.path as osp
import pickle
import time

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from lavila.data import datasets
from lavila.data.video_transforms import Permute
from lavila.models import models
from lavila.utils.preprocess import generate_tokenizer
from lavila.utils import distributed as dist_utils
from eval_narrator import decode_one


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return index, self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def get_args_parser():
    parser = argparse.ArgumentParser(description='lavila infer narrator', add_help=False)
    parser.add_argument('--dataset', default='ego4d', type=str, choices=['ego4d'])
    parser.add_argument('--root',
                        default='datasets/Ego4D/video_5min_chunks_288px/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata',
                        default='datasets/Ego4D/ego4d_train.pkl',
                        type=str, help='path to metadata file')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--use-half', action='store_true')
    parser.add_argument('--clip-length', default=4, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=16, type=int, help='clip stride')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--caption-sample', default='multinomial_sample',
                        choices=['multinomial_sample', 'beam_sample', 'group_beam_search'])
    parser.add_argument('--caption-top-k', default=None, type=int)
    parser.add_argument('--caption-top-p', default=0.95, type=float)
    parser.add_argument('--caption-num-beams', default=1, type=int)
    parser.add_argument('--caption-num-beam-groups', default=1, type=int)
    parser.add_argument('--caption-temperature', default=0.7, type=float)
    parser.add_argument('--caption-length-penalty', default=1.0, type=float)
    parser.add_argument('--caption-num-return-sequences', default=10, type=int)
    parser.add_argument('--caption-max-len', default=77, type=int)
    parser.add_argument('--caption-early-stop', action='store_true', help='early stopping to save computation')
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
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    return parser


def main(args):
    dist_utils.init_distributed_mode(args)
    print(args)

    if args.resume:
        ckpt_path = args.resume
    elif osp.isfile(osp.join(args.output_dir, 'checkpoint_best.pt')):
        ckpt_path = osp.join(args.output_dir, 'checkpoint_best.pt')
    else:
        raise Exception('no checkpoint found')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # create model
    old_args = ckpt['args']
    print('=> creating model: {}'.format(old_args.model))
    model = getattr(models, old_args.model)(
        text_use_cls_token=old_args.use_cls_token,
        gated_xattn=old_args.gated_xattn,
        timesformer_gated_xattn=old_args.timesformer_gated_xattn,
        num_frames=old_args.clip_length,
        drop_path_rate=0,
    )
    model.cuda()
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    torch.backends.cudnn.benchmark = True

    # Data loading
    print("=> creating dataset")
    tokenizer = generate_tokenizer(old_args.model)

    crop_size = 224 if '336PX' not in old_args.model else 336
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),  # T H W C -> C T H W
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in old_args.model else
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
    ])

    val_dataset = datasets.VideoCaptionDatasetCLIP(
        args.dataset,
        args.root,
        args.metadata,
        transform=val_transform,
        is_training=False,
        tokenizer=tokenizer,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        sparse_sample=False,
        subsample_stride=1,
    )
    val_dataset = IndexedDataset(val_dataset)

    print(len(val_dataset))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )
    print('len(val_loader) = {}'.format(len(val_loader)))

    model.eval()
    if args.use_half:
        model.half()

    id_offset = 0
    all_captions_cache = []
    end = time.time()
    with torch.no_grad():
        for data_iter, (indices, inputs) in enumerate(val_loader):
            indices = indices.tolist()
            if data_iter % args.print_freq == 0:
                print("finished {}/{} in {}".format(data_iter, len(val_loader), time.time() - end))
                end = time.time()
            if len(inputs) == 2 or len(inputs) == 3:
                images = inputs[0].cuda(non_blocking=True)
                if args.use_half:
                    images = images.half()

                image_features = dist_utils.get_model(model).encode_image(images)
                if not isinstance(image_features, (list, tuple)):
                    image_tokens = image_features
                else:
                    image_tokens = image_features[1]
                if args.caption_sample == 'multinomial_sample':
                    generated_text_ids, ppls = dist_utils.get_model(model).generate(
                        image_tokens,
                        tokenizer,
                        target=None,
                        max_text_length=args.caption_max_len,
                        top_k=args.caption_top_k,
                        top_p=args.caption_top_p,
                        num_return_sequences=args.caption_num_return_sequences,
                        temperature=args.caption_temperature,
                        early_stopping=args.caption_early_stop,
                    )
                elif args.caption_sample == 'beam_sample':
                    generated_text_ids, ppls = dist_utils.get_model(model).beam_sample(
                        image_tokens,
                        tokenizer,
                        target=None,
                        max_text_length=args.caption_max_len,
                        top_k=args.caption_top_k,
                        top_p=args.caption_top_p,
                        temperature=args.caption_temperature,
                        length_penalty=args.caption_length_penalty,
                        num_beams=args.caption_num_beams,
                        num_return_sequences=args.caption_num_return_sequences,
                    )
                elif args.caption_sample == 'group_beam_search':
                    assert args.caption_num_beam_groups > 1 and args.caption_num_beams % args.caption_num_beam_groups == 0
                    generated_text_ids, ppls = dist_utils.get_model(model).group_beam_search(
                        image_tokens,
                        tokenizer,
                        target=None,
                        max_text_length=args.caption_max_len,
                        top_k=args.caption_top_k,
                        top_p=args.caption_top_p,
                        temperature=args.caption_temperature,
                        length_penalty=args.caption_length_penalty,
                        num_beams=args.caption_num_beams,
                        num_beam_groups=args.caption_num_beam_groups,
                        num_return_sequences=args.caption_num_return_sequences,
                    )
                for j in range(generated_text_ids.shape[0] // args.caption_num_return_sequences):
                    generated_text_str_list = []
                    ppls_list = []
                    for k in range(args.caption_num_return_sequences):
                        jj = j * args.caption_num_return_sequences + k
                        generated_text_str = decode_one(generated_text_ids[jj], tokenizer)
                        generated_text_str_list.append(generated_text_str)
                        ppls_list.append(ppls[jj].item())
                    video_uid, t_start, t_end, _ = val_loader.dataset.dataset.samples[indices[j]]
                    if args.caption_num_return_sequences == 1:
                        all_captions_cache.append((video_uid, t_start, t_end, generated_text_str, ppls[jj].item()))
                    else:
                        all_captions_cache.append((video_uid, t_start, t_end, generated_text_str_list, ppls_list))
                id_offset += generated_text_ids.shape[0]

    pickle.dump(all_captions_cache, open(osp.join(args.output_dir, 'cache.{}.pkl'.format(args.rank)), 'wb'))

    torch.distributed.barrier()
    disorded_list = []
    total_num = 0
    if args.rank == 0:
        for i in range(args.world_size):
            print('=> reading {}'.format(osp.join(args.output_dir, f'cache.{i}.pkl')))
            sublist = pickle.load(open(osp.join(args.output_dir, f'cache.{i}.pkl'), 'rb'))
            disorded_list.append(sublist)
            total_num += len(sublist)
        ordered_list = []
        for i in range(total_num):
            ordered_list.append(disorded_list[i % args.world_size][i // args.world_size])
        print(f"{len(val_dataset)}/{len(ordered_list)}")
        ordered_list = ordered_list[:len(val_dataset)]
        pickle.dump(ordered_list, open(osp.join(args.output_dir, 'total.pkl'), 'wb'))
        for i in range(args.world_size):
            print('=> deleting {}'.format(osp.join(args.output_dir, f'cache.{i}.pkl')))
            os.remove(osp.join(args.output_dir, f'cache.{i}.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila infer narrator', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
