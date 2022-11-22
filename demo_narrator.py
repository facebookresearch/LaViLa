# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import urllib.request
from collections import OrderedDict

import decord
import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from lavila.data.video_transforms import Permute
from lavila.data.datasets import get_frame_ids, video_loader_by_frames
from lavila.models.models import VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL
from lavila.models.tokenizer import MyGPT2Tokenizer
from eval_narrator import decode_one


def main(args):

    vr = decord.VideoReader(args.video_path)
    num_seg = 4
    frame_ids = get_frame_ids(0, len(vr), num_segments=num_seg, jitter=False)
    frames = video_loader_by_frames('./', args.video_path, frame_ids)

    ckpt_name = 'vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth'
    ckpt_path = os.path.join('modelzoo/', ckpt_name)
    os.makedirs('modelzoo/', exist_ok=True)
    if not os.path.exists(ckpt_path):
        print('downloading model to {}'.format(ckpt_path))
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/{}'.format(ckpt_name), ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    # instantiate the model, and load the pre-trained weights
    model = VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
        text_use_cls_token=False,
        project_embed_dim=256,
        gated_xattn=True,
        timesformer_gated_xattn=False,
        freeze_lm_vclm=False,      # we use model.eval() anyway
        freeze_visual_vclm=False,  # we use model.eval() anyway
        num_frames=4,
        drop_path_rate=0.
    )
    model.load_state_dict(state_dict, strict=True)
    if args.cuda:
        model.cuda()
    model.eval()

    # transforms on input frames
    crop_size = 336
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
    ])
    frames = val_transform(frames)
    frames = frames.unsqueeze(0)  # fake a batch dimension

    tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)
    with torch.no_grad():
        if args.cuda:
            frames = frames.cuda(non_blocking=True)
        image_features = model.encode_image(frames)
        generated_text_ids, ppls = model.generate(
            image_features,
            tokenizer,
            target=None,  # free-form generation
            max_text_length=77,
            top_k=None,
            top_p=0.95,   # nucleus sampling
            num_return_sequences=10,  # number of candidates: 10
            temperature=0.7,
            early_stopping=True,
        )

    for i in range(10):
        generated_text_str = decode_one(generated_text_ids[i], tokenizer)
        print('{}: {}'.format(i, generated_text_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila narrator demo')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--video-path', default='assets/3c0dffd0-e38e-4643-bc48-d513943dc20b_012_014.mp4', type=str, help='video path')
    args = parser.parse_args()
    main(args)
