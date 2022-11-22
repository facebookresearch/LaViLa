# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, GPT2LMHeadModel

import lavila.models.loss as loss
from lavila.models.gpt2_gated import GPT2LMHeadModel as GatedGPT2LMHeadModel
from lavila.models.gpt2_gated import augment_gpt2_config
from lavila.models.narrator import VCLM_HF
from lavila.models.openai_clip import load as load_openai_clip
from lavila.models.openai_model import QuickGELU, Transformer
from lavila.models.timesformer import SpaceTimeTransformer
from lavila.models.utils import remap_keys, rsetattr


class VideoClassifier(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes: int,
                 **kwargs,
                 ):
        super().__init__()
        self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        self.fc_cls = nn.Linear(vision_model.num_features, num_classes, bias=True)

        self.fc_cls.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_cls.bias.data.zero_()

    def forward(self, image, use_checkpoint=False):
        image_embed = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        logit = self.fc_cls(self.dropout(image_embed))
        return logit


class VideoClassifierMultiHead(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes_list: list,
                 **kwargs,
                 ):
        super().__init__()
        self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        self.fc_cls = nn.ModuleList(
            [nn.Linear(vision_model.num_features, num_classes, bias=True) for num_classes in num_classes_list]
        )

        for m in self.fc_cls:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()

    def forward(self, image, use_checkpoint=False):
        image_embed = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        logit_list = [m(self.dropout(image_embed)) for m in self.fc_cls]
        return logit_list


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 tempearture_init=0.07,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)  # used to be `models.transformer.LayerNorm``

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        print("=> initialize initial temperature with {}".format(tempearture_init))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image, use_checkpoint=False, apply_project=True):
        x = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
        if not apply_project:
            return x
        x = x @ self.image_projection

        return x

    def encode_text(self, text, use_checkpoint=False):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, use_checkpoint=use_checkpoint)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, use_checkpoint=False, norm_embed=False):
        image_embed = self.encode_image(image, use_checkpoint=use_checkpoint)
        text_embed = self.encode_text(text, use_checkpoint=use_checkpoint)

        if norm_embed:
            image_embed = F.normalize(image_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}


class CLIP_HF(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 text_width: int,
                 text_model: nn.Module,
                 text_use_cls_token: bool,
                 text_is_regressive: bool,
                 tempearture_init=0.07,
                 **kwargs,
                 ):
        super().__init__()

        self.vision_width = vision_width
        self.visual = vision_model
        self.text_width = text_width
        self.textual = text_model
        self.text_use_cls_token = text_use_cls_token
        self.text_is_regressive = text_is_regressive

        if 'projection' not in kwargs:
            self.projection = 'default'
        else:
            self.projection = kwargs['projection']
        if self.projection == 'default':
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        elif self.projection == 'frozen_in_time':
            self.image_projection = nn.Sequential(
                nn.Linear(vision_width, embed_dim)
            )
            self.text_projection = nn.Sequential(
                nn.ReLU(),
                nn.Linear(text_width, embed_dim)
            )
        print("=> initialize initial temperature with {}".format(tempearture_init))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))

        self.initialize_parameters()

    def initialize_parameters(self):
        if self.projection == 'default':
            nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
            nn.init.normal_(self.text_projection, std=self.text_width ** -0.5)
        else:
            nn.init.normal_(self.image_projection[0].weight, std=self.vision_width ** -0.5)
            nn.init.normal_(self.text_projection[1].weight, std=self.text_width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image, use_checkpoint=False, apply_project=True):
        x = self.visual(image, use_checkpoint=use_checkpoint)
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
        if not apply_project:
            return x
        if self.projection == 'default':
            x = x @ self.image_projection
        else:
            x = self.image_projection(x)

        return x

    def encode_text(self, text, attention_mask=None, use_checkpoint=False):
        if use_checkpoint:
            if isinstance(self.textual, DistilBertModel):
                pass
                # print("DistilBertModel does not support gradient checkpointing. Skipping even if use_checkpoint=True")
            else:
                self.textual.gradient_checkpointing_enable()
        else:
            self.textual.gradient_checkpointing_disable()
        # text, attention_mask = text.squeeze(1), attention_mask.squeeze(1)
        # ^ uncomment this only when doing local debugging (distributed=False)
        x = self.textual(text, attention_mask=attention_mask)

        if self.text_is_regressive:
            # gpt-style
            x = x.last_hidden_state
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        else:
            # bert-style
            if self.text_use_cls_token:
                x = x.last_hidden_state
                x = x[torch.arange(x.shape[0]), 0, :]
            else:
                x = x.pooler_output
        if self.projection == 'default':
            x = x @ self.text_projection
        else:
            x = self.text_projection(x)

        return x

    def forward(self, image, text, mask=None, use_checkpoint=False, norm_embed=False):
        image_embed = self.encode_image(image, use_checkpoint=use_checkpoint)
        text_embed = self.encode_text(text, attention_mask=mask, use_checkpoint=use_checkpoint)

        if norm_embed:
            image_embed = F.normalize(image_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}


def get_loss(model, args, tokenizer=None):
    if model.startswith('CLIP'):
        return loss.CLIPLoss(
            use_vissl=args.contrastive_use_vissl,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
        )
    elif model.startswith('VCLM'):
        return loss.CaptionLoss(tokenizer=tokenizer)
    else:
        raise NotImplementedError


def get_metric_names(model):
    if model.startswith('CLIP'):
        return ['loss', 'clip_loss', 'clip_acc']
    elif model.startswith('VCLM'):
        return ['loss', 'caption_loss', 'caption_acc', 'ppl']
    else:
        raise NotImplementedError


def CLIP_OPENAI_TIMESFORMER_BASE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_width=768,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    model.transformer.load_state_dict(clip_model.transformer.state_dict())
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    return model


def CLIP_OPENAI_TIMESFORMER_LARGE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=224, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-L/14', 'cpu')
    print("=> Loading CLIP (ViT-L/14) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_width=1024,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    model.transformer.load_state_dict(clip_model.transformer.state_dict())
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    return model


def CLIP_OPENAI_TIMESFORMER_LARGE_336PX(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=336, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-L/14@336px', 'cpu')
    print("=> Loading CLIP (ViT-L/14@336px) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_width=1024,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    model.transformer.load_state_dict(clip_model.transformer.state_dict())
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    return model


def CLIP_OPENAI_TIMESFORMER_BASE_DISTILBERT_BASE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    text_model = DistilBertModel.from_pretrained(
        'distilbert-base-uncased',
    )
    kwargs.pop('text_use_cls_token')  # ignore args.use_cls_token since DistilBert does not have pooler on top
    model = CLIP_HF(
        embed_dim=project_embed_dim,
        vision_width=vision_model.embed_dim,
        vision_model=vision_model,
        text_width=768,
        text_model=text_model,
        text_use_cls_token=True,  # DistilBert does not have pooler on top
        text_is_regressive=False,
        tempearture_init=temperature_init,
        **kwargs,
    )

    return model


def CLIP_OPENAI_TIMESFORMER_LARGE_DISTILBERT_BASE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=224, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-L/14', 'cpu')
    print("=> Loading CLIP (ViT-L/14) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    text_model = DistilBertModel.from_pretrained(
        'distilbert-base-uncased',
    )
    kwargs.pop('text_use_cls_token')  # ignore args.use_cls_token since DistilBert does not have pooler on top
    model = CLIP_HF(
        embed_dim=project_embed_dim,
        vision_width=vision_model.embed_dim,
        vision_model=vision_model,
        text_width=768,
        text_model=text_model,
        text_use_cls_token=True,  # DistilBert does not have pooler on top
        text_is_regressive=False,
        tempearture_init=temperature_init,
        **kwargs,
    )

    return model


def CLIP_OPENAI_TIMESFORMER_LARGE_336PX_DISTILBERT_BASE(
    num_frames=4, timesformer_gated_xattn=False, drop_path_rate=0, timesformer_freeze_space=False,
    temperature_init=0.07, project_embed_dim=256, **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=336, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
        drop_path_rate=drop_path_rate,
    )
    clip_model, _ = load_openai_clip('ViT-L/14@336px', 'cpu')
    print("=> Loading CLIP (ViT-L/14@336px) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    if timesformer_freeze_space:
        print("=> Freeze the space part in TimeSformer")
        freeze_list, unfreeze_list = [], []
        for n, p in vision_model.named_parameters():
            if n not in remapped_state_dict or n == 'cls_token':
                p.requires_grad = True
                unfreeze_list.append(n)
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in TimeSformer: {}".format(freeze_list))
        print(" Learn the rest parts in TimeSformer: {}".format(unfreeze_list))

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    text_model = DistilBertModel.from_pretrained(
        'distilbert-base-uncased',
    )
    kwargs.pop('text_use_cls_token')  # ignore args.use_cls_token since DistilBert does not have pooler on top
    model = CLIP_HF(
        embed_dim=project_embed_dim,
        vision_width=vision_model.embed_dim,
        vision_model=vision_model,
        text_width=768,
        text_model=text_model,
        text_use_cls_token=True,  # DistilBert does not have pooler on top
        text_is_regressive=False,
        tempearture_init=temperature_init,
        **kwargs,
    )

    return model


def CLIP_HF_EGOVLP_DISTILBERT_BASE(num_frames=4, project_embed_dim=256, **kwargs):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
    )
    vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
    vision_model.load_state_dict(vit_model.state_dict(), strict=False)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    text_model = DistilBertModel.from_pretrained(
        'distilbert-base-uncased',
    )
    kwargs.pop('text_use_cls_token')  # ignore args.use_cls_token since DistilBert does not have pooler on top
    kwargs.update({'projection': 'frozen_in_time'})
    model = CLIP_HF(
        embed_dim=project_embed_dim,
        vision_width=vision_model.embed_dim,
        vision_model=vision_model,
        text_width=768,
        text_model=text_model,
        text_use_cls_token=True,  # DistilBert does not have pooler on top
        text_is_regressive=False,
        **kwargs,
    )

    return model


def CLIP_HF_TIMESFORMER_DISTILBERT_BASE(num_frames=4, drop_path_rate=0, temperature_init=0.07, project_embed_dim=256, **kwargs):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        drop_path_rate=drop_path_rate,
    )
    vit_model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True)
    vision_model.load_state_dict(vit_model.state_dict(), strict=False)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    text_model = DistilBertModel.from_pretrained(
        'distilbert-base-uncased',
    )
    kwargs.pop('text_use_cls_token')  # ignore args.use_cls_token since DistilBert does not have pooler on top
    model = CLIP_HF(
        embed_dim=project_embed_dim,
        vision_width=vision_model.embed_dim,
        vision_model=vision_model,
        text_width=768,
        text_model=text_model,
        text_use_cls_token=True,  # DistilBert does not have pooler on top
        text_is_regressive=False,
        tempearture_init=temperature_init,
        **kwargs,
    )

    return model


def VCLM_OPENAI_VITB16_GPT2_LARGE(gated_xattn=False, freeze_lm_vclm=False,
                                  freeze_visual_vclm=False, freeze_visual_vclm_temporal=False, **kwargs):
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    vision_model = clip_model.visual
    kwargs.pop('text_use_cls_token')

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-large",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=768,
        vision_model=vision_model,
        text_width=1280,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=20,
        **kwargs,
    )

    return model


def VCLM_OPENAI_VITB16_GPT2_XL(gated_xattn=False, freeze_lm_vclm=False,
                               freeze_visual_vclm=False, freeze_visual_vclm_temporal=False, **kwargs):
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    vision_model = clip_model.visual
    kwargs.pop('text_use_cls_token')

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=768,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def VCLM_OPENAI_VITL14_GPT2_XL(gated_xattn=False, freeze_lm_vclm=False,
                               freeze_visual_vclm=False, freeze_visual_vclm_temporal=False, **kwargs):
    clip_model, _ = load_openai_clip('ViT-L/14', 'cpu')
    vision_model = clip_model.visual
    kwargs.pop('text_use_cls_token')

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=1024,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def VCLM_OPENAI_VITL14_336PX_GPT2_XL(gated_xattn=False, freeze_lm_vclm=False,
                                     freeze_visual_vclm=False, freeze_visual_vclm_temporal=False, **kwargs):
    clip_model, _ = load_openai_clip('ViT-L/14@336px', 'cpu')
    vision_model = clip_model.visual
    kwargs.pop('text_use_cls_token')

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=1024,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def VCLM_OPENAI_TIMESFORMER_BASE_GPT2(
    gated_xattn=False,
    random_init_gpt2=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    timesformer_gated_xattn=False,
    **kwargs,
):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=1, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    if not random_init_gpt2:
        print('Loading LM from pretrained weights..')
        for n, p in gpt2.named_parameters():
            rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=768,
        vision_model=vision_model,
        text_width=768,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=12,
        **kwargs,
    )

    return model


def VCLM_OPENAI_TIMESFORMER_BASE_GPT2_XL(
    gated_xattn=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    timesformer_gated_xattn=False,
    **kwargs,
):
    vision_model = SpaceTimeTransformer(
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
    )
    clip_model, _ = load_openai_clip('ViT-B/16', 'cpu')
    print("=> Loading CLIP (ViT-B/16) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=12)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=768,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def VCLM_OPENAI_TIMESFORMER_LARGE_GPT2_XL(
    gated_xattn=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    timesformer_gated_xattn=False,
    **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=224, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
    )
    clip_model, _ = load_openai_clip('ViT-L/14', 'cpu')
    print("=> Loading CLIP (ViT-L/14x) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=2, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=1024,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def VCLM_OPENAI_TIMESFORMER_LARGE_GPT2(
    gated_xattn=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    timesformer_gated_xattn=False,
    **kwargs
):
    vision_model = SpaceTimeTransformer(
        img_size=224, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
    )
    clip_model, _ = load_openai_clip('ViT-L/14', 'cpu')
    print("=> Loading CLIP (ViT-L/14x) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=1, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=1024,
        vision_model=vision_model,
        text_width=768,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=12,
        **kwargs,
    )

    return model


def VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
    gated_xattn=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    timesformer_gated_xattn=False,
    **kwargs,
):
    vision_model = SpaceTimeTransformer(
        img_size=336, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,
        num_frames=num_frames,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=True,
        act_layer=QuickGELU,
        is_tanh_gating=timesformer_gated_xattn,
    )
    clip_model, _ = load_openai_clip('ViT-L/14@336px', 'cpu')
    print("=> Loading CLIP (ViT-L/14@336px) weights")
    remapped_state_dict = remap_keys(clip_model.visual.state_dict(), transformer_layers=24)
    res = vision_model.load_state_dict(remapped_state_dict, strict=False)
    print(res)
    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )
    new_config = augment_gpt2_config(gpt2.config, cross_attn_freq=3, gated_xattn=gated_xattn)
    text_decoder = GatedGPT2LMHeadModel(new_config)
    for n, p in gpt2.named_parameters():
        rsetattr(text_decoder, n + '.data', p.data)

    if freeze_lm_vclm:
        print('Freeze the LM part of TextDecoder of VCLM')
        text_decoder.freeze_lm_weights()

    if freeze_visual_vclm:
        print('Freeze the spatial part of VideoEncoder of VCLM')
        vision_model.freeze_spatial_weights()

    if freeze_visual_vclm_temporal:
        print('Freeze the temporal part of VideoEncoder of VCLM')
        vision_model.freeze_temporal_weights()

    model = VCLM_HF(
        vision_width=1024,
        vision_model=vision_model,
        text_width=1600,
        text_decoder=text_decoder,
        num_img_queries=256,
        dim_head=64,
        heads=25,
        **kwargs,
    )

    return model


def CLIP_OPENAI_VITB32(**kwargs):
    model, _ = load_openai_clip('ViT-B/32', 'cpu')
    return model


def CLIP_OPENAI_VITB16(**kwargs):
    model, _ = load_openai_clip('ViT-B/16', 'cpu')
    return model


def CLIP_OPENAI_VITL14(**kwargs):
    model, _ = load_openai_clip('ViT-L/14', 'cpu')
    return model


def CLIP_OPENAI_VITL14_336PX(**kwargs):
    model, _ = load_openai_clip('ViT-L/14@336px', 'cpu')
    return model
