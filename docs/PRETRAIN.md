# LAVILA Pretraining

In this doc, we provide a step-by-step guide (with commands) to train LaViLa.
Note that we recommend running the following job with four 8x V100 (32GB) nodes (or eight nodes for the larger backbone) using [submitit](https://github.com/facebookincubator/submitit).
See how to install submitit at [here](./MODEL_ZOO.md#multi-node-training).


## Pre-training Dual-Encoder Baseline

We first pre-train a dual-encoder baseline with human annotations on Ego4d clips.
The goal is (1) to establish a comparable baseline for LAVILA, and (2) provide a video encoder for narrator (see below).
We use a default batch size of 32 per gpu so that the total batch size for InfoNCE loss is `32*8*4=1024`.

<details><summary> Train a baseline dual-encoder (with TSF-B) </summary>

```bash
python run_with_submitit_pretrain.py --model CLIP_OPENAI_TIMESFORMER_BASE \
    --norm-embed --freeze-temperature \
    --fix-lr --contrastive-use-vissl \
    --nodes 4 --use_volta32
```
</details>

To fit a High-Resolution TimeSformer-Large with a sufficient batch size, we use [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert), a memory-efficient text encoder, instead of the original text encoder in the CLIP. Additionally we apply [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint.html) and [Zero Redundancy Optimizer (ZeRO)](https://arxiv.org/abs/1910.02054).

<details><summary> Train a baseline dual-encoder (with TSF-L@HR) </summary>

```bash
python run_with_submitit_pretrain.py --model CLIP_OPENAI_TIMESFORMER_LARGE_336PX_DISTILBERT_BASE \
    --batch-size 8 \
    --use-checkpoint --use-zero \
    --norm-embed --freeze-temperature \
    --fix-lr --contrastive-use-vissl \
    --nodes 8 --use_volta32
```
</details>

## Training and Evaluating Narrator

The narrator is a *visually conditioned* large language model (VCLM), which comprises a pre-trained video encoder (obtained above), a text decoder (GPT-2 family), and a few gated cross-attention modules that attends visual information while captioning. Both the video encoder and the text decoder are kept frozen while the cross-attention modules are learnable.

Note that we turn off Pytorch's automatic mixed-precision (AMP) during training the narrator. We observe training is instable if AMP is on.

Also note that `$PATH` can be found in the `Vis. Encoder` column of [MODEL_ZOO.md#Narrator](./MODEL_ZOO.md#narrator). If you are using your own checkpoint (e.g. pre-trained in the previous step), please make sure that the following keys in the checkpoint have been dropped: `epoch`, `optimizer`, and `scaler`.

<details><summary> Train a baseline narrator (TSF-B as visual encoder and GPT-2 base as textual decoder) </summary>

```bash
python run_with_submitit_pretrain.py \
    --model VCLM_OPENAI_TIMESFORMER_BASE_GPT2 \
    --gated-xattn --freeze-lm-vclm --freeze-visual-vclm --freeze-visual-vclm-temporal \
    --fix-lr --batch-size 8 --clip-grad-value 1.0 --eval-freq 1 --disable-amp \
    --nodes 4 --use_volta32 --resume $PATH   # Eg. $PATH can be "modelzoo/clip_openai_timesformer_base.baseline.ep_0003.pth"
```

</details>

<details><summary> Train a strong narrator (TSF-L@HR as visual encoder and GPT-2 XL as textual decoder) </summary>

```bash
python run_with_submitit_pretrain.py \
    --model VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL \
    --gated-xattn --freeze-lm-vclm --freeze-visual-vclm --freeze-visual-vclm-temporal --use-checkpoint \
    --fix-lr --batch-size 8 --clip-grad-value 1.0 --eval-freq 1 --disable-amp \
    --nodes 4 --use_volta32 --resume $PATH   # Eg. $PATH can be "modelzoo/clip_openai_timesformer_large_336px_distilbert_base.baseline.ep_0003.pth"
```
</details>

<details><summary> Evaluate the narrator on Ego4D val split </summary>

```bash
torchrun --nproc_per_node=1 eval_narrator.py \
    --caption-top-p 0.95 --caption-temperature 0.7 \
    --eval-freq 10000 \     # evaluate on the val split of Ego4D (1/10000-subset for fast evaluation)
    --resume $VCLM_CHECKPOINT
```
This will output some common NLG metrics, such as BLEU-x, METEOR, ROUGE_L, and CIDEr (using the human narrations as ground-truth).
</details>

## Narrating video clips using LAVILA-Narrator


<details><summary> Infer the narrator </summary>

```bash
python run_with_submitit_infer_narrator.py \
    --metadata datasets/Ego4D/ego4d_train.pkl \
    --batch-size 64 \
    --resume $PATH --use-half \
    --nodes 4 --use_volta32
```
</details>

It will generate a pickle file (`$output_dir/total.pkl`) which is a list of quintuples - `(video_uid: str, start_time: float, end_time: float, narration_list: List[str], NLL_list: List[float])`.

For narrator-generated narrations on Ego4D ground-truth clips, we also provide a [replica](https://dl.fbaipublicfiles.com/lavila/metadata/ego4d/ego4d_train.narrator_63690737.return_10.pkl). Note that the narrator used here is our best performing one.

In addition, we can apply this narrator over the entire video for temporally dense auto-narration. We provide a [replica](https://dl.fbaipublicfiles.com/lavila/metadata/ego4d/ego4d_train.uncovered_all.narrator_63690737.return_5.pkl) (excluding the annotated clips).

## Rephrasing human narrations using LAVILA-Rephraser

Rephraser is a standard LLM that can paraphrase narrations in existing clips.
Specifically, we use an off-the-shelf T5-based paraphraser which is publicly available at [Hugging Face's model hub](https://huggingface.co/ramsrigouthamg/t5-large-paraphraser-diverse-high-quality).
For more details, please refer to the [model card](https://huggingface.co/ramsrigouthamg/t5-large-paraphraser-diverse-high-quality).

For rephrased human narrations on Ego4D ground-truth clips, we provide a [replica](https://dl.fbaipublicfiles.com/lavila/metadata/ego4d/ego4d_train.rephraser.no_punkt_top3.pkl).


## Pre-training LAVILA Dual-Encoder
Now we are ready to pre-train our LAVILA's dual-encoder by combining human annotations (augmented by Rephraser) and the Narrator-generated narrations.

<details><summary> Training a LaViLa dual-encoder </summary>

```bash
python run_with_submitit_pretrain.py --model CLIP_OPENAI_TIMESFORMER_BASE \
    --metadata datasets/Ego4D/ego4d_train.rephraser.no_punkt_top3.pkl \
    --metadata-aux datasets/Ego4D/ego4d_train.narrator_63690737.return_10.pkl \  # also optionally add `datasets/Ego4D/ego4d_train.uncovered_all.narrator_63690737.return_5.pkl`
    --norm-embed --freeze-temperature \
    --freeze-pseudo-temperature \
    --fix-lr --contrastive-use-vissl \
    --nodes 4 --use_volta32
```
</details>

## Down-stream Evaluation
With the pre-trained dual-encoder at hand, we now can do zero-shot or fine-tuning evalution evaluations on down-stream benchmarks.
Please refer to [MODEL_ZOO.md](./MODEL_ZOO.md#zero-shot) for more details.
