# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of the code is from https://github.com/huggingface/transformers/blob/main/src/transformers/generation_utils.py
# Modified by Yue Zhao
# The original code is under Apache 2.0 License


import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import BeamSearchScorer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    TypicalLogitsWarper,
    LogitNormalization,
)

from lavila.models.coca import CrossAttention, LayerNorm
from lavila.models.openai_model import VisionTransformer
from lavila.models.timesformer import SpaceTimeTransformer


class VCLM_HF(nn.Module):
    def __init__(self,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 text_width: int,
                 text_decoder: nn.Module,
                 num_img_queries=256,
                 dim_head=64,
                 heads=8,
                 **kwargs,
                 ):
        super().__init__()
        self.vision_width = vision_width
        self.visual = vision_model
        self.text_width = text_width
        self.text_decoder = text_decoder

        self.img_queries = nn.Parameter(torch.empty(num_img_queries, text_width))
        self.img_attn_pool = CrossAttention(
            dim=text_width, context_dim=vision_width,
            dim_head=dim_head, heads=heads,
            norm_context=True
        )
        self.img_attn_pool_norm = LayerNorm(text_width)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.img_queries, std=self.text_width ** -0.5)

    def encode_image(self, image, use_checkpoint=False):
        if isinstance(self.visual, VisionTransformer):
            # openai_model.VisionTransformer accepts (N, C, H, W) instead of (N, C, T, H, W)
            image = image.permute(0, 2, 1, 3, 4)  # BCTHW -> BTCHW
            bb, tt, _, _, _ = image.shape
            x = self.visual(image.reshape(-1, *image.shape[2:]), use_checkpoint=use_checkpoint, cls_at_last=False)  # NLD
            x = x.view(bb, tt, *x.shape[1:])
            x = x.permute(0, 3, 1, 2)
        elif isinstance(self.visual, SpaceTimeTransformer):
            image = image.permute(0, 2, 1, 3, 4).contiguous()  # BCTHW -> BTCHW
            bb, tt, _, _, _ = image.shape
            x = self.visual.forward_features(image, use_checkpoint=use_checkpoint, cls_at_last=False)  # NLD
            x = x.permute(0, 2, 1)
        else:
            x = self.visual(image, use_checkpoint=use_checkpoint, mean_at_last=False)
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]

        x = x.flatten(start_dim=2)  # BDTHW -> BD(THW)
        x = x.permute(0, 2, 1)      # BDN -> BND
        img_queries = repeat(self.img_queries, 'n d -> b n d', b=x.shape[0])
        img_queries = self.img_attn_pool(img_queries, x)
        img_queries = self.img_attn_pool_norm(img_queries)
        return img_queries

    def forward(self, image, text, mask=None, use_checkpoint=False, norm_embed=False):
        if use_checkpoint:
            self.text_decoder.gradient_checkpointing_enable()
        else:
            self.text_decoder.gradient_checkpointing_disable()

        text, labels = text[:, :-1], text[:, 1:]
        # mask = mask[:, :-1]
        image_tokens = self.encode_image(image, use_checkpoint=use_checkpoint)

        output_decoder = self.text_decoder(text.contiguous(), encoder_hidden_states=image_tokens)
        text_tokens_logits = output_decoder.logits
        text_tokens_logits = rearrange(text_tokens_logits, 'b n c -> b c n')

        return {'text_tokens_logits': text_tokens_logits,
                'labels': labels}

    def generate(self, image_tokens, tokenizer, target=None, max_text_length=77, top_k=None, top_p=None,
                 num_return_sequences=1, temperature=1.0, teacher_forcing=False, early_stopping=False):
        image_tokens = image_tokens.repeat_interleave(num_return_sequences, dim=0)
        device = image_tokens.device
        generated_text_ids = torch.LongTensor([[tokenizer.bos_token_id]] * image_tokens.shape[0]).to(device)
        condition_text_ids = generated_text_ids.clone()

        logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, typical_p=None, temperature=temperature, num_beams=1)

        nlls, num_tokens = torch.zeros(image_tokens.shape[0]).to(device), torch.zeros(image_tokens.shape[0]).to(device)
        is_reach_eos = torch.zeros(image_tokens.shape[0]).bool().to(device)
        with torch.no_grad():
            for i in range(max_text_length - 1):
                output_decoder = self.text_decoder(condition_text_ids, encoder_hidden_states=image_tokens)
                decoded_token_logits = output_decoder.logits
                next_token_logits = decoded_token_logits[:, -1, :]
                if target is not None:
                    nll = F.cross_entropy(next_token_logits, target[:, i+1], ignore_index=tokenizer.pad_token_id, reduction='none')
                    nlls += nll
                    num_tokens += target[:, i+1].ne(tokenizer.pad_token_id)
                else:
                    nll = torch.special.entr(F.softmax(next_token_logits, dim=1)).sum(dim=1)
                    nlls += nll * (~is_reach_eos)
                    num_tokens += (~is_reach_eos)
                # filtered_p = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p, device=device)
                next_token_logits = logits_warper(generated_text_ids, next_token_logits)
                filtered_p = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(filtered_p, num_samples=1)
                is_reach_eos = is_reach_eos | (next_token[:, 0] == tokenizer.eos_token_id)
                if early_stopping and torch.all(is_reach_eos):
                    break

                if teacher_forcing:
                    condition_text_ids = target[:, :i+2]
                else:
                    condition_text_ids = torch.cat((generated_text_ids, next_token), dim=1)

                generated_text_ids = torch.cat((generated_text_ids, next_token), dim=1)
        if target is not None:
            return generated_text_ids, torch.exp(nlls / num_tokens)
        else:
            return generated_text_ids, torch.exp(nlls / num_tokens)

    def beam_sample(self, image_tokens, tokenizer, target=None, max_text_length=77, top_k=None, top_p=None,
                    temperature=1.0, length_penalty=1.,
                    num_beams=3, num_return_sequences=1, teacher_forcing=False, early_stopping=False):
        batch_size = image_tokens.shape[0]
        device = image_tokens.device
        input_ids = torch.ones((batch_size, 1), device=device, dtype=torch.long)
        input_ids = input_ids * tokenizer.bos_token_id

        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, num_beams * num_return_sequences).view(-1).to(device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        batch_beam_size, cur_len = input_ids.shape

        logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, typical_p=None, temperature=temperature, num_beams=num_beams)

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size * num_return_sequences, num_beams=num_beams,
            device=device,
            length_penalty=length_penalty,
        )
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        beam_scores = torch.zeros((batch_size, num_beams)).to(device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        is_reach_eos = torch.zeros(batch_beam_size).bool().to(device)
        with torch.no_grad():
            for i in range(max_text_length - 1):
                output_decoder = self.text_decoder(
                    input_ids,
                    encoder_hidden_states=image_tokens.repeat_interleave(num_beams * num_return_sequences, dim=0)
                )
                decoded_token_logits = output_decoder.logits
                next_token_logits = decoded_token_logits[:, -1, :]

                next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
                # supposed to be the line below, but ignore temporarily
                # next_token_scores_processed = logits_processor(input_ids, next_token_scores)
                next_token_scores_processed = next_token_scores
                next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
                # supposed to be the line below, but do a simple top_k+top_p temporarily
                next_token_scores = logits_warper(input_ids, next_token_scores)
                # next_token_scores = top_k_top_p_filtering(next_token_scores, top_k=top_k, top_p=top_p, device=device)

                vocab_size = next_token_scores.shape[-1]
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                probs = F.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                beam_outputs = beam_scorer.process(
                    input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

                is_reach_eos = is_reach_eos | (input_ids[:, -1] == tokenizer.eos_token_id)
                if beam_scorer.is_done or torch.all(is_reach_eos):
                    break

            sequence_outputs = beam_scorer.finalize(
                input_ids,
                beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_length=max_text_length,
            )

            sequences = sequence_outputs["sequences"]
            sequence_scores = sequence_outputs["sequence_scores"]
        return sequences, sequence_scores

    def group_beam_search(self, image_tokens, tokenizer, target=None, max_text_length=77, top_k=None, top_p=None,
                          temperature=1.0, length_penalty=1.,
                          num_beams=6, num_beam_groups=3,
                          num_return_sequences=1, teacher_forcing=False, early_stopping=False):
        batch_size = image_tokens.shape[0]
        device = image_tokens.device
        input_ids = torch.ones((batch_size, 1), device=device, dtype=torch.long)
        input_ids = input_ids * tokenizer.bos_token_id

        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        batch_beam_size, cur_len = input_ids.shape

        logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, typical_p=None, temperature=temperature, num_beams=num_beams)

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size, num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            num_beam_hyps_to_keep=num_return_sequences, device=device,
            length_penalty=length_penalty,
        )
        num_sub_beams = num_beams // num_beam_groups
        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        is_reach_eos = torch.zeros(batch_beam_size).bool().to(device)
        with torch.no_grad():

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            for i in range(max_text_length - 1):
                output_decoder = self.text_decoder(
                    input_ids,
                    encoder_hidden_states=image_tokens.repeat_interleave(num_beams, dim=0)
                )
                decoded_token_logits = output_decoder.logits

                for beam_group_idx in range(num_beam_groups):
                    group_start_idx = beam_group_idx * num_sub_beams
                    group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                    group_size = group_end_idx - group_start_idx

                    # indices of beams of current group among all sentences in batch
                    batch_group_indices = []

                    for batch_idx in range(batch_size):
                        batch_group_indices.extend(
                            [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                        )
                    group_input_ids = input_ids[batch_group_indices]

                    # select outputs of beams of current group only
                    next_token_logits = decoded_token_logits[batch_group_indices, -1, :]

                    next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
                    vocab_size = next_token_scores.shape[-1]

                    # supposed to be the line below, but ignore temporarily
                    # next_token_scores_processed = logits_processor(input_ids, next_token_scores)
                    next_token_scores_processed = next_token_scores
                    next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                    next_token_scores = next_token_scores.expand_as(next_token_scores_processed)
                    next_token_scores = logits_warper(input_ids, next_token_scores)
                    # next_token_scores = top_k_top_p_filtering(next_token_scores, top_k=top_k, top_p=top_p, device=device)

                    # reshape for beam search
                    next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                    next_token_scores, next_tokens = torch.topk(
                        next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                    )

                    next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                    next_tokens = next_tokens % vocab_size

                    # stateless
                    beam_outputs = beam_scorer.process(
                        group_input_ids,
                        next_token_scores,
                        next_tokens,
                        next_indices,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        beam_indices=None
                    )
                    beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                    beam_next_tokens = beam_outputs["next_beam_tokens"]
                    beam_idx = beam_outputs["next_beam_indices"]

                    input_ids[batch_group_indices] = group_input_ids[beam_idx]
                    group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                    current_tokens[batch_group_indices] = group_input_ids[:, -1]
                    reordering_indices[batch_group_indices] = (
                        num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (beam_idx % group_size)
                    )

                input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

                is_reach_eos = is_reach_eos | (input_ids[:, -1] == tokenizer.eos_token_id)
                if beam_scorer.is_done or torch.all(is_reach_eos):
                    break

            sequence_outputs = beam_scorer.finalize(
                input_ids,
                beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_length=max_text_length,
                beam_indices=None,
            )

            sequences = sequence_outputs["sequences"]
            sequence_scores = sequence_outputs["sequence_scores"]
        return sequences, sequence_scores

    def _get_logits_warper(
        self, top_k=None, top_p=None, typical_p=None,
        temperature=None, num_beams=None, renormalize_logits=None,
    ):
        top_k = top_k if top_k is not None else 0
        top_p = top_p if top_p is not None else 1.0
        typical_p = typical_p if typical_p is not None else 1.
        temperature = temperature if temperature is not None else 1.
        warpers = LogitsProcessorList()

        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if typical_p is not None and typical_p < 1.0:
            warpers.append(TypicalLogitsWarper(mass=typical_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        # `LogitNormalization` should always be the last logit processor, when present
        if renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers
