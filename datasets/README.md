# Preparing datasets for LAVILA

Please download the (selected) datasets from the official websites and place or sim-link them under `$LAVILA_ROOT/datasets/`.

```bash
$LAVILA_ROOT/datasets/
    CharadesEgo/
    EGTEA/
    EK100/
    Ego4D/
```

## Ego4D
1. Download [Ego4D videos](https://ego4d-data.org/docs/start-here/#download-data) (license is required).

2. Preprocess

    We cut each video into 5-minute-long chunks and resize the smaller size to be 288 pixels for faster IO. Please refer to [this script](scripts/crop_and_resize_ego4d.sh) for more details. 

3. Download annotations

    a. Download [egomcq.json](https://drive.google.com/file/d/1-5iRYf4BCHmj4MYQYFRMY4bhsWJUN3rW/view) to `$LAVILA_ROOT/datasets/Ego4D` (if you want to evaluate EgoMCQ).

    b. Download [metadata for train split](https://dl.fbaipublicfiles.com/lavila/metadata/ego4d/ego4d_train.pkl) and [val split](https://dl.fbaipublicfiles.com/lavila/metadata/ego4d/ego4d_val.pkl) to `$LAVILA_ROOT/datasets/Ego4D` ((if you want to train LAVILA from scratch).

The fold should look like this:
```bash
$LAVILA_ROOT/datasets/
    Ego4D/
        ego4d_train.pkl
        ego4d_val.pkl
        egomcq.json
        video_288px/
            000786a7-3f9d-4fe6-bfb3-045b368f7d44.mp4/
                0.mp4
                300.mp4
            000a3525-6c98-4650-aaab-be7d2c7b9402.mp4/
                0.mp4
            ...
```


## EPIC-Kitchens-100 (EK-100)

1. Download annotations

```bash
# Assume that you are under `datasets/EK100/`
git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations
```

2. Download videos.

    a. For raw videos, please download them from [https://epic-kitchens.github.io/](https://epic-kitchens.github.io/).

    b. (Recommended) The raw videos are huge (~1 TB). As an alternative, please check out a [resized version](https://utexas.box.com/s/l7ij81ie5q07p9fdg0vtejihq61liln9).

3. (For EK-100 MIR)

    a. Generate the relevancy matrix of train/val splits using [the official code](https://github.com/mwray/Joint-Part-of-Speech-Embeddings).

    b. (Recommended) The generated result has some randomness. Therefore, we also provide the [replica of train split](https://dl.fbaipublicfiles.com/lavila/metadata/EK100/caption_relevancy_EPIC_100_retrieval_train.pkl) and [val split](https://dl.fbaipublicfiles.com/lavila/metadata/EK100/caption_relevancy_EPIC_100_retrieval_test.pkl). Please put them to the folder `$LAVILA_ROOT/datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/`.


The folder should look like this:
```bash
$LAVILA_ROOT/datasets/
    EK100/
        epic-kitchens-100-annotations/
            EPIC_100_train.csv
            EPIC_100_validation.csv
            ...
            retrieval_annotations/relevancy/  # this appears if you do 3.
                caption_relevancy_EPIC_100_retrieval_train.pkl
                caption_relevancy_EPIC_100_retrieval_test.pkl
        video_ht256px/
            P01/
                P01_01.MP4
                P01_02.MP4
                ...
                P01_19.MP4
            P02/
                P02_01.MP4
                P02_02.MP4
                ...
                P02_15.MP4
            ...
```

## CharadesEgo

1. Download annotations at [https://prior.allenai.org/projects/charades-ego](https://prior.allenai.org/projects/charades-ego).
```bash
### Annotations
# Assume that you are under `datasets/CharadesEgo/`
wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/CharadesEgo.zip
unzip CharadesEgo.zip && rm CharadesEgo.zip
```

2. Download data (~11GB) at [https://prior.allenai.org/projects/charades-ego](https://prior.allenai.org/projects/charades-ego).
```bash
### Data
wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/CharadesEgo_v1_480.tar
tar -xvf CharadesEgo_v1_480.tar  # Or specify an external path using `-C` and sim-link it to here
rm CharadesEgo_v1_480.tar
```

3. (For fine-tuning CharadesEgo) Download two additional metadata files: [clip-level metadata (train)](https://dl.fbaipublicfiles.com/lavila/metadata/CharadesEgo/metadata_filtered_train.pkl) and [clip-level metadata (val)](https://dl.fbaipublicfiles.com/lavila/metadata/CharadesEgo/metadata_filtered_val.pkl). Put them to the folder `$LAVILA_ROOT/datasets/CharadesEgo/CharadesEgo/`.

The folder should look like this:
```bash
$LAVILA_ROOT/datasets/
    CharadesEgo/
        CharadesEgo/
            CharadesEgo_v1_train_only1st.csv
            CharadesEgo_v1_test_only1st.csv
            ...
            metadata_filtered_train.pkl  # this appears if you do 3.
            metadata_filtered_val.pkl    # this appears if you do 3.
        CharadesEgo_v1_480/
            005BU.mp4
            005BUEGO.mp4
            ...
```


## EGTEA

1. Visit [https://cbs.ic.gatech.edu/fpv/](https://cbs.ic.gatech.edu/fpv/).

2. Download `TRIMMED_ACTION_CLIPS` (~20GB) and `ACTION_ANNOTATIONS` and untar to the current folder `$LAVILA_ROOT/datasets/EGTEA`.

```bash
unzip action_annotation.zip -d EGTEA/ && rm action_annotation.zip
```

The folder should look like this:
```bash
$LAVILA_ROOT/datasets/
    EGTEA/
        train_split1.txt
        test_split1.txt
        cropped_clips/
            OP01-R01-PastaSalad/
                OP01-R01-PastaSalad-1002316-1004005-F024051-F024101.mp4
                OP01-R01-PastaSalad-1004110-1021110-F024057-F024548.mp4
                OP01-R01-PastaSalad-1022590-1024050-F024539-F024581.mp4
                ...
            OP01-R02-TurkeySandwich/
                OP01-R02-TurkeySandwich-102320-105110-F002449-F002529.mp4
                OP01-R02-TurkeySandwich-105440-106460-F002528-F002558.mp4
                OP01-R02-TurkeySandwich-107332-133184-F002513-F003259.mp4
                ...
            ...
```
