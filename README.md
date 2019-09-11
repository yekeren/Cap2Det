# Cap2Det
Implementation of our ICCV 2019 paper "Cap2Det: Learning to AmplifyWeak Caption Supervision for Object Detection".

If you found this repository useful, please cite our paper

```
@article{DBLP:journals/corr/abs-1907-10164,
  author    = {Keren Ye and
               Mingda Zhang and
               Adriana Kovashka and
               Wei Li and
               Danfeng Qin and
               Jesse Berent},
  title     = {Cap2Det: Learning to Amplify Weak Caption Supervision for Object Detection},
  journal   = {CoRR},
  volume    = {abs/1907.10164},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.10164},
  archivePrefix = {arXiv},
  eprint    = {1907.10164},
  timestamp = {Thu, 01 Aug 2019 08:59:33 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1907-10164},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# Preparing data.

We provide scripts to preprocess datasets such as MSCOCO 2017, Flick30K, VOC 2007/2012.

Preparing these training data involves three steps:

*  Extract region proposals using the Selective Search algorithm.
*  Encode the annotations and the region proposals to the tfrecord files.
*  Gather the open vocabulary, extract required word embeddings from pre-trained GloVe model.

## Pascal VOC

The Pascal VOC datasets are used for:

*  Validating the Weakly Supervised Object Detection (WSOD) models.
*  Validating our Cap2Det models.

The datasets do not have captions annotations.

For the first goal (WSOD), we tested our models on both VOC2007 and VOC2012.
we train on 5,011 and 11,540 trainval images respectively,
We evaluate on 4,952 and 10,991 test images.

For the second goal (Cap2Det),
we train models on MSCOCO or Flickr30k, then evaluate on the 4,952 test images in VOC2007.

### Extract region proposals

```
python "tools/create_pascal_selective_search_data.py" \
  --logtostderr \
  --data_dir="${DATA_DIR}" \
  --year="${YEAR}" \
  --set="${SET}" \
  --output_dir="${OUTPUT_DIR}"
```

### Generate tfrecord files

```
python "tools/create_pascal_tf_record.py" \
  --logtostderr \
  --data_dir="${DATA_DIR}" \
  --year="${YEAR}" \
  --set="${SET}" \
  --output_path="${OUTPUT_PATH}" \
  --label_map_path="${LABEL_MAP_PATH}" \
  --proposal_data_path="${PROPOSAL_DATA_PATH}" \
  --ignore_difficult_instances
```

### All-in-one

Putting all together, one can just run the following all-in-one command.
It shall create a new raw-data-voc directory, and generate files in it.

```
sh scripts/prepare_voc.sh "raw-data-voc"
```

## MSCOCO 2017

We use the 591,435 annotated captions paired to the 118,287 train2017 images for training.

### Extract region proposals

```
python "tools/create_coco_selective_search_data.py" \
  --logtostderr \
  --train_image_file="${TRAIN_IMAGE_FILE}" \
  --val_image_file="${VAL_IMAGE_FILE}" \
  --test_image_file="${TEST_IMAGE_FILE}" \
  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
  --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}"
```

### Generate tfrecord files.

```
python "tools/create_coco_tf_record.py" \
  --logtostderr \
  --train_image_file="${TRAIN_IMAGE_FILE}" \
  --val_image_file="${VAL_IMAGE_FILE}" \
  --test_image_file="${TEST_IMAGE_FILE}" \
  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
  --train_caption_annotations_file="${TRAIN_CAPTION_ANNOTATIONS_FILE}" \
  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
  --val_caption_annotations_file="${VAL_CAPTION_ANNOTATIONS_FILE}" \
  --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
  --proposal_data="${PROPOSAL_DATA}" \
  --output_dir="${OUTPUT_DIR}"
```

### Gather the open vocabulary.

```
python "tools/create_coco_vocab.py" \
  --logtostderr \
  --train_caption_annotations_file="${TRAIN_CAPTION_ANNOTATIONS_FILE}" \
  --glove_file="${GLOVE_FILE}" \
  --output_vocabulary_file="${OUTPUT_VOCABULARY_FILE}" \
  --output_vocabulary_word_embedding_file="${OUTPUT_VOCABULARY_WORD_EMBEDDING_FILE}" \
  --min_word_freq=${MIN_WORD_FREQ}
```

### All-in-one

Putting all together, one can just run the following all-in-one command.
It shall create a new raw-data-coco directory, and generate files in it.

```
sh scripts/prepare_coco.sh "raw-data-coco/"
```

## Flickr30K

## Image Ads
