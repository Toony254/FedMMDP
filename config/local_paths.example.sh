#!/usr/bin/env bash
# Copy this file to config/local_paths.sh and fill in local values.
# The copied file is ignored by git and can safely store machine-specific paths.

export CONDA_BIN="conda"
export ENV_NAME="fedmmdp"

export FEDMMDP_COCO_ROOT="/path/to/COCO"
export FEDMMDP_FLICKR30K_ROOT="/path/to/flickr30k/flickr30k-images"

export FEDMMDP_IMAGENET_ROOT="/path/to/imagenet/domain_datasets"
export FEDMMDP_IAPR_ROOT="/path/to/iapr/domain_datasets"
export FEDMMDP_IMAGENET_SIGLIP_ROOT="/path/to/imagenet/domain_datasets"
export FEDMMDP_IAPR_SIGLIP_ROOT="/path/to/iapr_siglip/domain_datasets"

export FEDMMDP_IAPR_CLIP_ROOT="/path/to/preprocessed_iapr"
export FEDMMDP_IAPR_SIGLIP_PREPROCESS_ROOT="/path/to/preprocessed_iapr_siglip"
export FEDMMDP_TEXT_DATA_ROOT="/path/to/text_data"
export FEDMMDP_CLIENT_ARTIFACT_DIR="./artifacts/yClient"
