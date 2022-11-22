# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import functools
import torch
import torch.nn.functional as F


def inflate_positional_embeds(
    current_model_state_dict, new_state_dict,
    num_frames=4,
    load_temporal_fix='bilinear',
):
    # allow loading of timesformer with fewer num_frames
    curr_keys = list(current_model_state_dict.keys())
    if 'visual.temporal_embed' in new_state_dict and 'visual.temporal_embed' in curr_keys:
        load_temporal_embed = new_state_dict['visual.temporal_embed']
        load_num_frames = load_temporal_embed.shape[1]
        curr_num_frames = num_frames
        embed_dim = load_temporal_embed.shape[2]

        if load_num_frames != curr_num_frames:
            if load_num_frames > curr_num_frames:
                print(f'### loaded SpaceTimeTransformer model has MORE frames than current...'
                      f'### loading weights, filling in the extras via {load_temporal_fix}')
                new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
            else:
                print(f'### loaded SpaceTimeTransformer model has FEWER frames than current...'
                      f'### loading weights, filling in the extras via {load_temporal_fix}')
                if load_temporal_fix == 'zeros':
                    new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                    new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                elif load_temporal_fix in ['interp', 'bilinear']:
                    # interpolate
                    # unsqueeze so pytorch thinks its an image
                    mode = 'nearest'
                    if load_temporal_fix == 'bilinear':
                        mode = 'bilinear'
                    load_temporal_embed = load_temporal_embed.unsqueeze(0)
                    new_temporal_embed = F.interpolate(load_temporal_embed,
                                                       (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                else:
                    raise NotImplementedError
            new_state_dict['visual.temporal_embed'] = new_temporal_embed
    # allow loading with smaller spatial patches. assumes custom border crop, to append the
    # border patches to the input sequence
    if 'visual.pos_embed' in new_state_dict and 'visual.pos_embed' in curr_keys:
        load_pos_embed = new_state_dict['visual.pos_embed']
        load_num_patches = load_pos_embed.shape[1]
        curr_pos_embed = current_model_state_dict['visual.pos_embed']
        if load_num_patches != curr_pos_embed.shape[1]:
            raise NotImplementedError(
                'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

    return new_state_dict


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


# util functions to convert CLIP-style model keys to TimeSformer-style
def remap_keys(clip_state_dict, transformer_layers=12):
    remapped_state_dict = OrderedDict()
    key_mapping = {
        "class_embedding": "cls_token",
        "positional_embedding": "pos_embed",
        "conv1.weight": "patch_embed.proj.weight",
        "ln_pre.weight": "ln_pre.weight",
        "ln_pre.bias": "ln_pre.bias",
        "ln_post.weight": "norm.weight",
        "ln_post.bias": "norm.bias",
    }
    for layer in range(transformer_layers):
        key_mapping[f"transformer.resblocks.{layer}.attn.in_proj_weight"] = f"blocks.{layer}.attn.qkv.weight"
        key_mapping[f"transformer.resblocks.{layer}.attn.in_proj_bias"] = f"blocks.{layer}.attn.qkv.bias"
        key_mapping[f"transformer.resblocks.{layer}.attn.out_proj.weight"] = f"blocks.{layer}.attn.proj.weight"
        key_mapping[f"transformer.resblocks.{layer}.attn.out_proj.bias"] = f"blocks.{layer}.attn.proj.bias"
        key_mapping[f"transformer.resblocks.{layer}.ln_1.weight"] = f"blocks.{layer}.norm1.weight"
        key_mapping[f"transformer.resblocks.{layer}.ln_1.bias"] = f"blocks.{layer}.norm1.bias"
        key_mapping[f"transformer.resblocks.{layer}.mlp.c_fc.weight"] = f"blocks.{layer}.mlp.fc1.weight"
        key_mapping[f"transformer.resblocks.{layer}.mlp.c_fc.bias"] = f"blocks.{layer}.mlp.fc1.bias"
        key_mapping[f"transformer.resblocks.{layer}.mlp.c_proj.weight"] = f"blocks.{layer}.mlp.fc2.weight"
        key_mapping[f"transformer.resblocks.{layer}.mlp.c_proj.bias"] = f"blocks.{layer}.mlp.fc2.bias"
        key_mapping[f"transformer.resblocks.{layer}.ln_2.weight"] = f"blocks.{layer}.norm2.weight"
        key_mapping[f"transformer.resblocks.{layer}.ln_2.bias"] = f"blocks.{layer}.norm2.bias"

    for key in clip_state_dict:
        if key == 'proj':
            continue  # due to possible dim mismatch, we load this later
        if key == "class_embedding":
            clip_state_dict[key] = clip_state_dict[key].unsqueeze(0).unsqueeze(0)
        if key == "positional_embedding":
            clip_state_dict[key] = clip_state_dict[key].unsqueeze(0)
        remapped_state_dict[key_mapping[key]] = clip_state_dict[key]

    return remapped_state_dict
