# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import glob
import json
import numpy as np
import os.path as osp
import pickle
import random

import decord
import pandas as pd
import torch


def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def video_loader(root, vid, second, end_second=None, chunk_len=300, fps=30, clip_length=32, jitter=False):
    if chunk_len == -1:
        vr = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid)))
        second_offset = second
        if end_second is not None:
            end_second = min(end_second, len(vr) / vr.get_avg_fps())
        else:
            end_second = len(vr) / vr.get_avg_fps()
    else:
        chunk_start = int(second) // chunk_len * chunk_len
        second_offset = second - chunk_start
        vr = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start)))
    if fps == -1:
        fps = vr.get_avg_fps()

    # calculate frame_ids
    frame_offset = int(np.round(second_offset * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)
    if chunk_len == -1:
        if end_second <= second:
            raise ValueError("end_second should be greater than second")
        else:
            frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)
    else:
        frame_ids = get_frame_ids(frame_offset, frame_offset + total_duration, num_segments=clip_length, jitter=jitter)

    # load frames
    if max(frame_ids) < len(vr):
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    else:
        # find the remaining frames in the next chunk
        try:
            frame_ids_part1 = list(filter(lambda frame_id: frame_id < len(vr), frame_ids))
            frames_part1 = vr.get_batch(frame_ids_part1).asnumpy()
            vr2 = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start + chunk_len)))
            frame_ids_part2 = list(filter(lambda frame_id: frame_id >= len(vr), frame_ids))
            frame_ids_part2 = [min(frame_id % len(vr), len(vr2) - 1) for frame_id in frame_ids_part2]
            frames_part2 = vr2.get_batch(frame_ids_part2).asnumpy()
            frames = np.concatenate([frames_part1, frames_part2], axis=0)
        # the next chunk does not exist; the current chunk is the last one
        except (RuntimeError, decord.DECORDError) as error:
            print(error)
            frame_ids = get_frame_ids(min(frame_offset, len(vr) - 1), len(vr), num_segments=clip_length, jitter=jitter)
            frames = vr.get_batch(frame_ids).asnumpy()

    frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    return torch.stack(frames, dim=0)


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        end = min(end, end_frame)
        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq


def video_loader_by_frames(root, vid, frame_ids):
    vr = decord.VideoReader(osp.join(root, vid))
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)


class VideoCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata, is_trimmed=True):
        self.dataset = dataset
        self.root = root
        self.is_trimmed = is_trimmed

        if self.dataset == 'ego4d':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset == 'ego4d_mcq':
            with open(metadata, 'r') as f:
                self.samples = json.load(f)
        elif self.dataset in ['ek100_cls', 'ek100_mir']:
            video_list = glob.glob(osp.join(self.root, '*/*.MP4'))
            fps_dict = {video: decord.VideoReader(video).get_avg_fps() for video in video_list}
            self.samples = []
            with open(metadata) as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)  # skip the header
                for row in csv_reader:
                    pid, vid = row[1:3]
                    # start_frame, end_frame = int(row[6]), int(row[7])
                    # Deprecated: some videos might have fps mismatch issue
                    start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                    narration = row[8]
                    verb, noun = int(row[10]), int(row[12])
                    vid_path = '{}/{}.MP4'.format(pid, vid)
                    fps = fps_dict[osp.join(self.root, vid_path)]
                    start_frame = int(np.round(fps * start_timestamp))
                    end_frame = int(np.ceil(fps * end_timestamp))
                    self.samples.append((vid_path, start_frame, end_frame, narration, verb, noun))
            if self.dataset == 'ek100_mir':
                self.metadata_sentence = pd.read_csv(metadata[:metadata.index('.csv')] + '_sentence.csv')
                if 'train' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_train.pkl'), 'rb'))
                elif 'test' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_test.pkl'), 'rb'))
                else:
                    raise ValueError('{} should contain either "train" or "test"!'.format(metadata))
                self.relevancy = .1
        elif self.dataset == 'egtea':
            video_list = glob.glob(osp.join(self.root, '*/*'))
            len_dict = {video: len(decord.VideoReader(video)) for video in video_list}

            vn_list, labels = [], []
            for row in open(osp.join(osp.dirname(metadata), 'action_idx.txt')):
                row = row.strip()
                vn = int(row.split(' ')[-1])
                vn_list.append(vn)
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
            mapping_act2narration = {vn: narration for vn, narration in zip(vn_list, labels)}

            self.samples = []
            with open(metadata) as f:
                for row in f:
                    clip_id, action_idx = row.strip().split(' ')[:2]
                    video_id = '-'.join(clip_id.split('-')[:3])
                    vid_relpath = osp.join(video_id, '{}.mp4'.format(clip_id))
                    vid_fullpath = osp.join(self.root, video_id, '{}.mp4'.format(clip_id))
                    self.samples.append((vid_relpath, 0, len_dict[vid_fullpath], mapping_act2narration[int(action_idx)]))
        elif self.dataset == 'charades_ego':
            video_list = glob.glob(osp.join(self.root, '*.mp4'))
            fps_dict = {video: decord.VideoReader(video).get_avg_fps() for video in video_list}
            self.samples = []
            with open(metadata) as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)  # skip the header
                for row in csv_reader:
                    video_id = row[0]
                    if self.is_trimmed:
                        for action_tuple in row[9].split(';'):
                            if not action_tuple:
                                continue
                            action, start_timestamp, end_timestamp = action_tuple.split(' ')
                            start_timestamp, end_timestamp = float(start_timestamp), float(end_timestamp)
                            vid_path = '{}.mp4'.format(video_id)
                            fps = fps_dict[osp.join(self.root, vid_path)]
                            start_frame = int(np.round(fps * start_timestamp))
                            end_frame = int(np.ceil(fps * end_timestamp))
                            self.samples.append((vid_path, start_frame, end_frame, action))
                    else:
                        if not row[9]:
                            action_list = []
                        else:
                            action_list = [action_tuple.split(' ')[0] for action_tuple in row[9].split(';')]
                        vid_path = '{}.mp4'.format(video_id)
                        fps = fps_dict[osp.join(self.root, vid_path)]
                        duration = fps * float(row[10])
                        self.samples.append((vid_path, 0, duration, action_list))
        elif self.dataset == 'charades_ego_trimmed':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            raise NotImplementedError

    def get_raw_item(self, i, is_training=True, num_clips=1, clip_length=32, clip_stride=2, sparse_sample=False,
                     narration_selection='random'):
        if self.dataset == 'ego4d':
            if len(self.samples[i]) == 4:
                vid, start_second, end_second, narration = self.samples[i]
                frames = video_loader(self.root, vid, start_second,
                                      end_second=end_second,
                                      clip_length=clip_length,
                                      jitter=is_training)
                if isinstance(narration, list):
                    if narration_selection == 'random':
                        narration = random.choice(narration)
                    elif narration_selection == 'concat':
                        narration = '. '.join(narration)
                    elif narration_selection == 'list':
                        narration = narration
                    else:
                        raise ValueError
                return frames, narration
            elif len(self.samples[i]) == 5:
                # TODO: need better filtering strategy based on nll
                vid, start_second, end_second, narration, _ = self.samples[i]
                frames = video_loader(self.root, vid, start_second,
                                      end_second=end_second,
                                      clip_length=clip_length,
                                      jitter=is_training)
                if isinstance(narration, list):
                    if narration_selection == 'random':
                        narration = random.choice(narration)
                    elif narration_selection == 'concat':
                        narration = '. '.join(narration)
                    elif narration_selection == 'list':
                        narration = narration
                    else:
                        raise ValueError
                return frames, narration
        elif self.dataset == 'ego4d_mcq':
            itemMCQ = self.samples[str(i)]
            answerIndex = itemMCQ['answer']
            textQuery = itemMCQ['query']['clip_text']
            sampleOptions = itemMCQ['choices']
            frames_options = []
            narration_options = []
            for option_id in range(len(sampleOptions)):
                option = sampleOptions[str(option_id)]
                frames = video_loader(self.root, option['video_uid'],
                                      float(option['clip_start']), end_second=float(option['clip_end']),
                                      clip_length=clip_length,
                                      jitter=is_training)
                frames_options.append(frames)
                narration_options.append(option['clip_text'])
            return textQuery, frames_options, narration_options, answerIndex, itemMCQ['types']
        elif self.dataset == 'ek100_mir':
            vid_path, start_frame, end_frame, narration, verb, noun = self.samples[i]
            # from third_party.EgoVLP.base.base_dataset import sample_frames_start_end
            # frame_ids = sample_frames_start_end(clip_length, start_frame, end_frame, sample='uniform', fix_start=None)
            frame_ids = get_frame_ids(start_frame, end_frame, num_segments=clip_length, jitter=is_training)
            frames = video_loader_by_frames(self.root, vid_path, frame_ids)
            if is_training:
                positive_list = np.where(self.relevancy_mat[i] > self.relevancy)[0].tolist()
                if positive_list != []:
                    pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                    if pos < len(self.metadata_sentence) and pos < self.relevancy_mat.shape[1]:
                        return frames, (self.metadata_sentence.iloc[pos][1], self.relevancy_mat[i][pos])
            else:
                return frames, (narration, 1)
        elif self.dataset == 'ek100_cls':
            vid_path, start_frame, end_frame, narration, verb, noun = self.samples[i]
            frame_ids = get_frame_ids(start_frame, end_frame, num_segments=clip_length, jitter=is_training)
            frames = video_loader_by_frames(self.root, vid_path, frame_ids)
            return frames, '{}:{}'.format(verb, noun)
        elif self.dataset == 'egtea':
            vid_path, start_frame, end_frame, sentence = self.samples[i]
            if is_training:
                assert num_clips == 1
                if end_frame < clip_length * clip_stride:
                    frames = video_loader_by_frames(self.root, vid_path, list(np.arange(0, end_frame)))
                    zeros = torch.zeros((clip_length * clip_stride - end_frame, *frames.shape[1:]))
                    frames = torch.cat((frames, zeros), dim=0)
                    frames = frames[::clip_stride]
                else:
                    start_id = np.random.randint(0, end_frame - clip_length * clip_stride + 1)
                    frame_ids = np.arange(start_id, start_id + clip_length * clip_stride, clip_stride)
                    frames = video_loader_by_frames(self.root, vid_path, frame_ids)
            else:
                if end_frame < clip_length * clip_stride:
                    frames = video_loader_by_frames(self.root, vid_path, list(np.arange(0, end_frame)))
                    zeros = torch.zeros((clip_length * clip_stride - end_frame, *frames.shape[1:]))
                    frames = torch.cat((frames, zeros), dim=0)
                    frames = frames[::clip_stride]
                    frames = frames.repeat(num_clips, 1, 1, 1)
                else:
                    frame_ids = []
                    for start_id in np.linspace(0, end_frame - clip_length * clip_stride, num_clips, dtype=int):
                        frame_ids.extend(np.arange(start_id, start_id + clip_length * clip_stride, clip_stride))
                    frames = video_loader_by_frames(self.root, vid_path, frame_ids)
            return frames, sentence
        elif self.dataset == 'charades_ego':
            vid_path, start_frame, end_frame, action_list = self.samples[i]
            if sparse_sample:
                frame_ids = get_frame_ids(start_frame, end_frame, num_segments=num_clips * clip_length, jitter=is_training)
                frames = video_loader_by_frames(self.root, vid_path, frame_ids)
            else:
                if end_frame < clip_length * clip_stride:
                    frames = video_loader_by_frames(self.root, vid_path, list(np.arange(0, end_frame)))
                    zeros = torch.zeros((clip_length * clip_stride - end_frame, *frames.shape[1:]))
                    frames = torch.cat((frames, zeros), dim=0)
                    frames = frames[::clip_stride]
                    frames = frames.repeat(num_clips, 1, 1, 1)
                else:
                    frame_ids = []
                    for start_id in np.linspace(0, end_frame - clip_length * clip_stride, num_clips, dtype=int):
                        frame_ids.extend(np.arange(start_id, start_id + clip_length * clip_stride, clip_stride))
                    print('frame_ids:', frame_ids)
                    frames = video_loader_by_frames(self.root, vid_path, frame_ids)
            return frames, action_list
        elif self.dataset == 'charades_ego_trimmed':
            vid, start_second, end_second, narration = self.samples[i]
            frames = video_loader(self.root, vid, start_second,
                                  end_second=end_second,
                                  chunk_len=-1,  # no chunk for CharadesEgo
                                  fps=-1,  # could be variable fps
                                  clip_length=clip_length,
                                  jitter=is_training)
            return frames, narration
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class VideoCaptionDatasetCLIP(VideoCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None,
                 is_training=True, tokenizer=None,
                 clip_length=32, clip_stride=2, sparse_sample=False,
                 narration_selection='random',
                 num_hard_negatives=0,
                 subsample_stride=None):
        super().__init__(dataset, root, metadata)

        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.sparse_sample = sparse_sample
        self.narration_selection = narration_selection
        self.num_hard_negatives = num_hard_negatives
        if num_hard_negatives > 0:
            assert self.dataset == 'htm_aa'

    def __getitem__(self, i):
        frames, caption = self.get_raw_item(
            i, is_training=self.is_training,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            sparse_sample=self.sparse_sample,
            narration_selection=self.narration_selection,
        )

        # ek100_mir will also output relevancy value
        if isinstance(caption, tuple):
            caption, relevancy = caption
        else:
            relevancy = 0.

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        if isinstance(caption, tuple):
            caption, mask = caption
            return frames, caption, mask, relevancy
        else:
            return frames, caption, relevancy


class VideoCaptionDatasetMCQ(VideoCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None,
                 is_training=True, tokenizer=None,
                 clip_length=32, clip_stride=2, sparse_sample=False,
                 narration_selection='random'):
        super().__init__(dataset, root, metadata)

        self.full_samples = self.samples.copy()
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.sparse_sample = sparse_sample
        self.narration_selection = narration_selection

    def __getitem__(self, i):

        textQuery, frames_options, narration_options, answerIndex, q_type = self.get_raw_item(
            i, is_training=self.is_training,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            sparse_sample=self.sparse_sample,
            narration_selection=self.narration_selection,
        )

        # apply transformation
        if self.transform is not None:
            frames_options = [self.transform(frames) for frames in frames_options]

        # tokenize caption
        if self.tokenizer is not None:
            textQuery = self.tokenizer(textQuery)
            narration_options = self.tokenizer(narration_options)
            if isinstance(textQuery, tuple):
                textQuery, mask_query = textQuery
                narration_options, mask_options = narration_options
                return (
                    textQuery, torch.stack(frames_options, dim=0),
                    narration_options, answerIndex, q_type,
                    mask_query, mask_options
                )
            else:
                return textQuery, torch.stack(frames_options, dim=0), narration_options, answerIndex, q_type


class VideoClassyDataset(VideoCaptionDatasetBase):
    def __init__(
        self, dataset, root, metadata, transform=None,
        is_training=True, label_mapping=None,
        num_clips=1,
        clip_length=32, clip_stride=2,
        sparse_sample=False,
        is_trimmed=True,
    ):
        super().__init__(dataset, root, metadata, is_trimmed=is_trimmed)

        self.transform = transform
        self.is_training = is_training
        self.label_mapping = label_mapping
        self.num_clips = num_clips
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.sparse_sample = sparse_sample

    def __getitem__(self, i):
        frames, label = self.get_raw_item(
            i, is_training=self.is_training,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            sparse_sample=self.sparse_sample,
        )

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.
                label = res_array
            else:
                label = self.label_mapping[label]

        return frames, label


def get_dataset(train_transform, tokenizer, args, is_training=True):
    if 'narration_selection' not in args:
        args.narration_selection = 'random'
    if args.model.startswith('CLIP') or args.model.startswith('VCLM'):
        return VideoCaptionDatasetCLIP(
            args.dataset, args.root, args.metadata, train_transform,
            is_training=is_training,
            tokenizer=tokenizer,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            sparse_sample=args.sparse_sample,
            narration_selection=args.narration_selection,
            num_hard_negatives=args.num_hard_neg if 'num_hard_neg' in args else 0,
        )
    else:
        raise NotImplementedError


def get_downstream_dataset(transform, tokenizer, args, subset='train', label_mapping=None):
    if subset == 'train':
        return VideoClassyDataset(
            args.dataset, args.root, args.metadata_train, transform,
            is_training=True, label_mapping=label_mapping,
            num_clips=args.num_clips,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            sparse_sample=args.sparse_sample,
        )
    elif subset == 'val':
        return VideoClassyDataset(
            args.dataset, args.root, args.metadata_val, transform,
            is_training=False, label_mapping=label_mapping,
            num_clips=args.num_clips,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            sparse_sample=args.sparse_sample,
            is_trimmed=not args.dataset == 'charades_ego'
        )
    else:
        assert ValueError("subset should be either 'train' or 'val'")
