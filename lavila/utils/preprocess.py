# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv

from lavila.models.tokenizer import MyBertTokenizer, MyDistilBertTokenizer, MyGPT2Tokenizer, SimpleTokenizer


def generate_label_map(dataset):
    if dataset == 'ek100_cls':
        print("Preprocess ek100 action label space")
        vn_list = []
        mapping_vn2narration = {}
        for f in [
            'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv',
            'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv',
        ]:
            csv_reader = csv.reader(open(f))
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                narration = row[8]
                if vn not in vn_list:
                    vn_list.append(vn)
                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)
                # mapping_vn2narration[vn] = [narration]
        vn_list = sorted(vn_list)
        print('# of action= {}'.format(len(vn_list)))
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
        print(labels[:5])
    elif dataset == 'charades_ego':
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        with open('datasets/CharadesEgo/CharadesEgo/Charades_v1_classes.txt') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print(labels[:5])
    elif dataset == 'egtea':
        print("=> preprocessing egtea action label space")
        labels = []
        with open('datasets/EGTEA/action_idx.txt') as f:
            for row in f:
                row = row.strip()
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
        print(len(labels), labels[:5])
    else:
        raise NotImplementedError
    return labels, mapping_vn2act


def generate_tokenizer(model):
    if model.endswith('DISTILBERT_BASE'):
        tokenizer = MyDistilBertTokenizer('distilbert-base-uncased')
    elif model.endswith('BERT_BASE'):
        tokenizer = MyBertTokenizer('bert-base-uncased')
    elif model.endswith('BERT_LARGE'):
        tokenizer = MyBertTokenizer('bert-large-uncased')
    elif model.endswith('GPT2'):
        tokenizer = MyGPT2Tokenizer('gpt2', add_bos=True)
    elif model.endswith('GPT2_MEDIUM'):
        tokenizer = MyGPT2Tokenizer('gpt2-medium', add_bos=True)
    elif model.endswith('GPT2_LARGE'):
        tokenizer = MyGPT2Tokenizer('gpt2-large', add_bos=True)
    elif model.endswith('GPT2_XL'):
        tokenizer = MyGPT2Tokenizer('gpt2-xl', add_bos=True)
    else:
        print("Using SimpleTokenizer because of model '{}'. "
              "Please check if this is what you want".format(model))
        tokenizer = SimpleTokenizer()
    return tokenizer
