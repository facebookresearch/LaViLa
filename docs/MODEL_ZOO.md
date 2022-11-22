# LAVILA Model Zoo

## Multi-node Training
We use multi-node training on a SLURM cluster with [submitit](https://github.com/facebookincubator/submitit) for producing the results and models in the paper.
Please install `submitit` in your conda environment:
```bash
pip install submitit
```


## Pre-training

Please refer to [PRETRAIN.md](./PRETRAIN.md).


## Narrator

| Visual Encoder | Text Decoder | METEOR | ROUGE-L | CIDEr | Pre-trained<br>Vis. Encoder (md5) | checkpoint (md5) |
| :------------: | :----------: | :----: | :-----: | :---: | :-------------------------------: | :--------: |
|     TSF-B      |    GPT-2     |  0.282 |  0.517  | 0.833 | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.baseline.ep_0003.pth) (dbcc4d)                        | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/vclm_openai_timesformer_base_gpt2_base.pt_ego4d.jobid_319630.ep_0002.md5sum_68a71f.pth) (68a71f)      |
|     TSF-L@HR   |    GPT-2 XL  |  0.298 |  0.539  | 0.977 | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_large_336px_distilbert_base.baseline.ep_0003.pth) (5c69b8) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth) (443263) |


<details><summary>Ego4D val split</summary>
<p>

```bash
torchrun --nproc_per_node=1 \
    eval_narrator.py \
    --caption-top-p 0.95 --caption-temperature 0.7 \
    --eval-freq 10000 \
    --resume $CHECKPOINT
```

</p></details>

## Zero-shot

<div class="table-wrapper" markdown="block">

|              | Backbone | EK-100 MIR<br>avg. mAP | EK-100 MIR<br>avg. nDCG | Charades-Ego<br>mAP^ | EGTEA<br> mean acc. | EgoMCQ<br>intra-video acc. |  checkpoint  |
| :----------: | :------: | :--------------------: | :---------------------: | :------------------: | :-----------------: | :------------------------: | :----------: |
| Prev. SOTA^^ |  TSF-B   |       22.1/23.3        |       22.1/27.9         |        25.2          |       17.6          |            57.2            | [Epoch 1](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/egovlp_epo1_converted_f16.md5sum_7a3d3b.pth), [best epoch](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/egovlp_converted_f16.md5sum_c33363.pth) |
|   LAVILA     |  TSF-B   |       29.7/30.9        |       31.5/32.0         |        26.8          |       28.9          |            59.9            | [Epoch 1](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.narrator_rephraser.ep_0001.md5sum_02dbb9.pth)^, [Epoch 5](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth) |
|   LAVILA     |  TSF-L   |       35.0/36.1        |       34.2/34.6         |        28.9          |       34.1          |            63.1            | [Epoch 1](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_large.narrator_rephraser.ep_0001.md5sum_9a25de.pth)^, [Epoch 3](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth) |

</div>

^ Note that the pre-trained checkpoint to evaluate CharadesEgo is different from that to evalute other datasets.
Specifically, we use the checkpoint at epoch 1 to zero-shot evaluate CharadesEgo and the checkpoint that achieves best average mAP on EK-100 MIR to evaluate other datasets, as is done in [EgoVLP](https://arxiv.org/pdf/2206.01670.pdf).
Our guess is that since CharadesEgo videos (captured by head-mounted mobile cameras) are visually different from Ego4D/EPIC-Kitchens videos (captured by professional action cameras, eg GoPro), pre-training on Ego4D videos for longer will lead to some potential domain discrepancy.

^^ We use the checkpoints released by [EgoVLP](https://github.com/showlab/EgoVLP) and convert them to be compatible with this codebase. Also note that our reproduced numbers are better than the reported numbers, especially on EK-100 MIR since we evaluate on raw videos directly (for more details, check out Appendix F & Table 10 in our paper).

<details><summary>1. EK-100 MIR</summary>
<p>

```bash
python eval_zeroshot.py --dataset ek100_mir --root datasets/EK100/video_ht256px/ --clip-length 4 --resume $PATH
```
By increasing the number of frames per clip, eg `--clip-length 16`, you are expected to see a better performance.

</p></details>

<details><summary>2. EK-100 CLS</summary>
<p>

```bash
python eval_zeroshot.py --dataset ek100_cls --metadata-val datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv  --resume $PATH 
```

</p></details>

<details><summary>3. Charades-Ego</summary>
<p>

```bash
python eval_zeroshot.py --dataset charades_ego --metadata-val datasets/CharadesEgo/CharadesEgo/CharadesEgo_v1_test_only1st.csv --root datasets/CharadesEgo/CharadesEgo_v1_480/ --clip-length 16 --sparse-sample --resume $PATH
```

</p></details>

<details><summary>4. EGTEA</summary>
<p>

```bash
python eval_zeroshot.py --dataset egtea --metadata-val datasets/EGTEA/test_split1.txt --root datasets/EGTEA/cropped_clips/ --clip-length 16 --clip-stride 2 --num-crops 3 --num-clips 10 --resume $PATH
```

</p></details>

<details><summary>5. EgoMCQ</summary>
<p>

```bash
python eval_zeroshot.py --dataset ego4d_mcq --metadata-val datasets/Ego4D/egomcq.json --root datasets/Ego4D/video_5min_chunks_288px/ --clip-length 4 --resume $PATH --use-half -j 4
```

</p></details>

## Fine-tuned

### EK-100 MIR

<div class="table-wrapper" markdown="block">

|        | Backbone | avg mAP | avg nDCG |   Pretrain (md5)   | Fine-tuned checkpoint | training log |
| :----: | :-------:| :-----: | :------: | :----------: | :-------------------: | :----------: |
| LAVILA |   TSF-B  |  50.5   |   65.0   | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth) (d73a9c) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_mir/clip_openai_timesformer_base.ft_ek100_mir.ep_0085.md5sum_c67d95.pth) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_mir/clip_openai_timesformer_base.ft_ek100_mir.jobid_57361.log) |
| LAVILA |   TSF-L  |  50.9   |   66.5   | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth) (c89337) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_mir/clip_openai_timesformer_large.ft_ek100_mir.ep_0095.md5sum_bd508b.pth) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_mir/clip_openai_timesformer_large.ft_ek100_mir.jobid_56606.log) |

</div>


<details><summary>Training and evaluating scripts</summary>
<p>

### Multi-node training (Slurm)
```bash
# TimeSformer-Base
python run_with_submitit_finetune_retrieval.py \
    --pretrain-model $PATH \
    --use-checkpoint --nodes 4

# TimeSformer-Large
python run_with_submitit_finetune_retrieval.py \
    --pretrain-model $PATH \
    --batch-size 4 \ 
    --use-checkpoint --nodes 4
```

### Single-machine training
```bash
torchrun --nproc_per_node=8 \
    main_finetune_retrieval.py \
    --output-dir $OUT_DIR \
    --pretrain-model $PATH \
    --use-checkpoint
```

Note that you might see a slight drop of performance when training on a single node compared to multiple nodes (everything else being the same) because of a smaller total batch size.

### Evaluation

Evaluation is done every `--eval-freq 5` epochs by default during fine-tuning.
If you want to evaluate any checkpoint after fine-tuning, please switch to `--evaluate` mode and specify the path to the checkpoint by `--resume $FINETUNED_CHECKPOINT`.
```bash
torchrun --nproc_per_node=1 \
    main_finetune_retrieval.py \
    --output-dir $OUT_DIR \
    --pretrain-model $PATH \
    --use-checkpoint \
    --evaluate \
    --resume $FINETUNED_CHECKPOINT
```


</p></details>

### CharadesEgo

<div class="table-wrapper" markdown="block">

|        | Backbone | video mAP |Pretrain^ (md5) |  Fine-tuned checkpoint | training log |
| :----: | :-------:| :------: | :-------: | :-------------------: | :----------: |
| LAVILA |   TSF-B  |    33.7   | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.narrator_rephraser.ep_0001.md5sum_02dbb9.pth) (02dbb9) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/charades_ego/clip_openai_timesformer_base.ft_charades_ego.ep_0005.md5sum_39bf4b.pth) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/charades_ego/clip_openai_timesformer_base.ft_charades_ego.jobid_65760.log) |
| LAVILA |   TSF-L  |    36.1   | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_large.narrator_rephraser.ep_0001.md5sum_9a25de.pth) (9a25de) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/charades_ego/clip_openai_timesformer_large.ft_charades_ego.ep_0003.md5sum_9448b2.pth) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/charades_ego/clip_openai_timesformer_large.ft_charades_ego.jobid_65760.log) |

</div>

^ Note that the pre-trained checkpoint for fine-tuning CharadesEgo is different from that for fine-tuning EK-100 or EGTEA. Same reason stated above.

<details><summary>Training and evaluating scripts</summary>
<p>

### Multi-node training (Slurm)

```bash
# TimeSformer-Base
python run_with_submitit_finetune_retrieval.py \
    --dataset charades_ego \
    --metadata datasets/CharadesEgo/CharadesEgo/metadata_filtered_train.pkl \
    --metadata-val datasets/CharadesEgo/CharadesEgo/CharadesEgo_v1_test_only1st.csv \
    --root datasets/CharadesEgo/CharadesEgo_v1_480/ \
    --epochs 10 \
    --save-freq 1 --eval-freq 1 \
    --sparse-sample \
    --pretrain-model $PATH \
    --use-checkpoint --nodes 4

# TimeSformer-Large
python run_with_submitit_finetune_retrieval.py \
    --dataset charades_ego \
    --metadata datasets/CharadesEgo/CharadesEgo/metadata_filtered_train.pkl \
    --metadata-val datasets/CharadesEgo/CharadesEgo/CharadesEgo_v1_test_only1st.csv \
    --root datasets/CharadesEgo/CharadesEgo_v1_480/ \
    --epochs 10 \
    --save-freq 1 --eval-freq 1 \
    --sparse-sample \
    --pretrain-model $PATH \
    --batch-size 4 \
    --use-checkpoint --nodes 4
```

### Evaluation
```bash
torchrun --nproc_per_node=1 \
    main_finetune_retrieval.py \
    --dataset charades_ego \
    --metadata datasets/CharadesEgo/CharadesEgo/metadata_filtered_train.pkl \
    --metadata-val datasets/CharadesEgo/CharadesEgo/CharadesEgo_v1_test_only1st.csv \
    --root datasets/CharadesEgo/CharadesEgo_v1_480/ \
    --output-dir $OUT_DIR \
    --sparse-sample \
    --pretrain-model $PATH \
    --evaluate \
    --resume $FINETUNED_CHECKPOINT
```

</p></details>

### EK-100 CLS

<div class="table-wrapper" markdown="block">

|        | Backbone | V+N+A multi-head | Verb top-1 | Noun top-1 | Action top-1 | Pretrain (md5) | Fine-tuned checkpoint | training log |
| :----: | :-------:| :--------------: | :--------: | :--------: |  :---------: | :------------: | :-------------------: | :----------: |
| LAVILA |   TSF-B  |       no         |    67.7    |    56.7    |    46.2      | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth) (d73a9c) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_cls/clip_openai_timesformer_base.ft_ek100_cls.single_head.ep_0100.md5sum_e8aa0c.pth) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_cls/clip_openai_timesformer_base.ft_ek100_cls.single_head.jobid_73363.log) |
| LAVILA |   TSF-B  |      yes         |    69.0    |    58.4    |    46.9      | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth) (d73a9c) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_cls/clip_openai_timesformer_base.ft_ek100_cls.ep_0100.md5sum_4e3575.pth) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_cls/clip_openai_timesformer_base.ft_ek100_cls.jobid_73361.log) |
| LAVILA |   TSF-L  |      yes         |    72.0    |    62.9   |     51.0      | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth) (c89337) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_cls/clip_openai_timesformer_large.ft_ek100_cls.ep_0090.md5sum_4a2509.pth) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ek100_cls/clip_openai_timesformer_large.ft_ek100_cls.jobid_74016.log) |
</div>

<details><summary>Training and evaluating scripts</summary>
<p>

### Multi-node training (Slurm)

```bash
# TimeSformer-Base
python run_with_submitit_finetune_classification.py \
    --pretrain-model $PATH \
    --use-vn-classifier --num-classes 97 300 3806 \
    --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
    --use-checkpoint --node 1

# TimeSformer-Large
python run_with_submitit_finetune_classification.py \
    --pretrain-model $PATH \
    --use-vn-classifier --num-classes 97 300 3806 \
    --use-sgd --wd 4e-5 --lr-multiplier-on-backbone 0.1 \
    --use-checkpoint --node 4
```

</p></details>

### EGTEA

<div class="table-wrapper" markdown="block">

|        | Backbone | mean Acc. | Pretrain (md5) | Fine-tuned checkpoint | training log |
| :----: | :-------:| :-------: | :------: | :-------------------: | :----------: |
| LAVILA |   TSF-B  |   70.12    | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth) (d73a9c) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/egtea/clip_openai_timesformer_base.ft_egtea.ep_0090.md5sum_3b1faf.pth) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/egtea/clip_openai_timesformer_base.ft_egtea.jobid_73358.log) |
| LAVILA |   TSF-L  |   76.00   | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth) (c89337) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/egtea/clip_openai_timesformer_large.ft_egtea.ep_0095.md5sum_a5ba17.pth) | [download](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/egtea/clip_openai_timesformer_large.ft_egtea.jobid_74026.log) |

</div>

<details><summary>Training and evaluating scripts</summary>
<p>

```bash
# TimeSformer-Base
python run_with_submitit_finetune_classification.py \
    --dataset egtea \
    --metadata-train datasets/EGTEA/train_split1.txt \
    --metadata-val datasets/EGTEA/test_split1.txt \
    --root datasets/EGTEA/cropped_clips/ \
    --pretrain-model $PATH \
    --num-classes 106 \
    --use-sgd --wd 4e-5 \
    --use-checkpoint --node 1

# TimeSformer-Large
python run_with_submitit_finetune_classification.py \
    --dataset egtea \
    --metadata-train datasets/EGTEA/train_split1.txt \
    --metadata-val datasets/EGTEA/test_split1.txt \
    --root datasets/EGTEA/cropped_clips/ \
    --pretrain-model $PATH \
    --num-classes 106 \
    --use-sgd --wd 4e-5 \
    --batch-size 4 \
    --use-checkpoint --node 4
```
### Evaluation
```bash
torchrun --nproc_per_node=1 \
    main_finetune_classification.py \
    --dataset egtea \
    --metadata-train datasets/EGTEA/train_split1.txt \
    --metadata-val datasets/EGTEA/test_split1.txt \
    --root datasets/EGTEA/cropped_clips/ \
    --output-dir $OUT_DIR \
    --pretrain-model $PATH \
    --num-classes 106 \
    --use-sgd --wd 4e-5 \
    --evaluate \
    --resume $FINETUNED_CHECKPOINT \
    --num-crops 3 --num-clips 10 \
    --use-half
```
</p></details>
