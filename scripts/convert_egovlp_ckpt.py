# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Usage:
```bash
PYTHONPATH=<EgoVLP-ROOT> python scripts/convert_egovlp_ckpt.py \
    --input-ckpt <EGOVLP_PATH> \
    --output-ckpt egovlp_converted.pth
```
'''

import argparse
from collections import OrderedDict
import torch


def get_args_parser():
    parser = argparse.ArgumentParser(description='Convert EgoVLP checkpoint', add_help=False)
    parser.add_argument('--input-ckpt', type=str)
    parser.add_argument('--output-ckpt', type=str)
    return parser


def main(args):
    input_ckpt = torch.load(args.input_ckpt, map_location='cpu')
    input_ckpt = input_ckpt['state_dict']
    output_ckpt = OrderedDict()
    for k in input_ckpt:
        if k.startswith('module.video_model'):
            output_ckpt[k.replace('module.video_model', 'module.visual')] = input_ckpt[k]
        elif k.startswith('module.text_model'):
            output_ckpt[k.replace('module.text_model', 'module.textual')] = input_ckpt[k]
        elif k.startswith('module.txt_proj'):
            output_ckpt[k.replace('module.txt_proj', 'module.text_projection')] = input_ckpt[k]
        elif k.startswith('module.vid_proj'):
            output_ckpt[k.replace('module.vid_proj', 'module.image_projection')] = input_ckpt[k]
        else:
            print(k)
            raise ValueError
    torch.save({
        'epoch': 0,
        'state_dict': output_ckpt,
        'best_acc1': 0,
        }, args.output_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert EgoVLP checkpoint', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
