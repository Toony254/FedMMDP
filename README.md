# FedMMDP-MASA

统一版 `FedMMDP-MASA` 项目，包含：

- 原始 `FedMMDP`
- 对比算法 `FedAvg`、`FedProx`、`MOON`、`Harmony`、`MASA`、`FedMEMA`
- 额外基线 `RawCLIP`、`CenterTraining`
- 公开数据蒸馏类方法 `FedMD`、`FedDF`、`Cream`

当前统一训练入口为 `src/main.py`。

## 1. 环境安装

```bash
conda create -n FedMMDP python=3.10 -y
conda activate FedMMDP
pip install -r requirements.txt
pip install finch-clust
```

说明：

- `FINCH` 包在 PyPI 上的安装名是 `finch-clust`。
- Windows 环境下项目内已带本地 `apex` 兼容 shim，不需要额外安装 NVIDIA Apex。

## 2. 路径约定

下面的命令默认你先配置好路径变量：

```bash
export CUDA_VISIBLE_DEVICES=0

# COCO 2017: 投影头预训练使用
export COCO2017_ROOT=/path/to/COCO

# MSCOCO 2014: FedMD / FedDF / Cream 使用
export MSCOCO2014_ROOT=/path/to/MSCOCO/2014

# IAPR 原始处理目录
export IAPR_RAW_DIR=./data/iapr_tc12

# 预处理后的联邦数据目录
export IMAGENET_CLIP_ROOT=./preprocessed_imagenet/domain_datasets
export IMAGENET_SIGLIP_ROOT=./preprocessed_imagenet/domain_datasets/siglip
export IAPR_CLIP_ROOT=./preprocessed_iapr/domain_datasets
# 本机数据集位置
export IMAGENET_CLIP_ROOT=/home/bd/data/zs/FedMMDP/preprocessed_imagenet/domain_datasets
export IMAGENET_SIGLIP_ROOT=/home/bd/data/zs/FedMMDP/preprocessed_imagenet/domain_datasets/siglip
export IAPR_CLIP_ROOT=/home/bd/data/zs/FedMMDP/preprocessed_iapr/domain_datasets


# 如果你已经准备好了 SigLIP 版 IAPR 特征，可使用该目录
export IAPR_SIGLIP_ROOT=./preprocessed_iapr/domain_datasets/siglip
```

建议在切换数据集或切换 `CLIP/SigLIP` 之前清理划分缓存：

```bash
rm -rf data_partition/*
```

## 3. 数据集预处理

### 3.1 ImageNet + CLIP

```bash
python src/datasets/preprocess_imagenet.py
```

默认输出到：

```bash
./preprocessed_imagenet/domain_datasets
```

### 3.2 ImageNet + SigLIP

如果你直接从原始 HuggingFace ImageNet Recaption 数据编码：

```bash
python src/datasets/preprocess_imagenet_siglip.py
```

如果你已经先生成了 `raw` 版按域数据，也可以从 `raw` 数据重新编码成 SigLIP：

```bash
python src/datasets/preprocess_imagenet_siglip_from_raw.py
```

默认输出到：

```bash
./preprocessed_imagenet/domain_datasets/siglip
```

### 3.3 IAPR_TC-12 原始数据整理

先下载并整理 IAPR_TC-12 原始数据：

```bash
python src/datasets/preprocess_IAPR_TC12.py \
  --save_dir "$IAPR_RAW_DIR" \
  --data_dir /path/to/IAPR_TC12_raw
```

如果脚本自行下载数据，可省略 `--data_dir`。

### 3.4 IAPR + CLIP

```bash
python src/datasets/preprocess_iapr.py
```

默认读取：

```bash
./data/iapr_tc12/train.json
./data/iapr_tc12/validation.json
```

默认输出到：

```bash
./preprocessed_iapr/domain_datasets
```

### 3.5 IAPR + SigLIP

当前仓库已经完整提供：

- `ImageNet + SigLIP` 预处理脚本
- `IAPR + CLIP` 预处理脚本

但没有单独提供 `preprocess_iapr_siglip.py`。因此，若你要运行 `SigLIP + IAPR` 对比实验，需要先准备与 `IAPR_CLIP_ROOT` 相同字段结构的 SigLIP 特征数据目录：

```bash
$IAPR_SIGLIP_ROOT/domain_dataset_0/{train,test}
$IAPR_SIGLIP_ROOT/domain_dataset_1/{train,test}
...
```

字段格式需与现有联邦数据保持一致：

- `id`
- `class_id`
- `processed_img`
- `cap_tokens`
- `domain_id`

后续训练命令已经按该目录格式给出。

## 4. 投影头预训练

### 4.1 CLIP 投影头预训练

该脚本使用 COCO 2017 图文对：

```bash
python src/networks/train_projector.py \
  --save_path saved/projector_weights \
  --lr 1e-4 \
  --epochs 10 \
  --batch_size 32 \
  --projector bottleneck \
  --coco_json "$COCO2017_ROOT/annotations/captions_train2017.json" \
  --coco_img_dir "$COCO2017_ROOT/train2017" \
  --coco_val_json "$COCO2017_ROOT/annotations/captions_val2017.json" \
  --coco_val_img_dir "$COCO2017_ROOT/val2017"
```

### 4.2 SigLIP 投影头预训练

```bash
python src/networks/train_projector_SigLIP.py \
  --save_path saved/projector_weights \
  --lr 1e-4 \
  --epochs 10 \
  --batch_size 32 \
  --projector mlp+norm \
  --model base \
  --coco_json "$COCO2017_ROOT/annotations/captions_train2017.json" \
  --coco_img_dir "$COCO2017_ROOT/train2017" \
  --coco_val_json "$COCO2017_ROOT/annotations/captions_val2017.json" \
  --coco_val_img_dir "$COCO2017_ROOT/val2017"
```

### 4.3 IAPR 专用 CLIP 投影头预训练

如果你希望在 IAPR_TC-12 上单独预训练 CLIP 投影头：

```bash
python src/networks/train_projector_IAPR_TC12.py \
  --iapr_data_dir "$IAPR_RAW_DIR" \
  --save_path saved/projector_weights \
  --lr 1e-4 \
  --epochs 20 \
  --batch_size 32 \
  --projector mlp+norm \
  --model RN50
```

`SigLIP` 没有单独的 IAPR 专用投影头脚本时，可直接复用 `train_projector_SigLIP.py` 训练得到的投影头。

## 5. FedMMDP 训练命令

说明：

- `--model` 必须与预编码数据使用的 backbone 保持一致：`clip` / `align` / `siglip` / `resnet`。
- `clip`、`align`、`siglip` 在运行期共用同一套 embedding projector 封装，但命令参数不能混写。
- `CLIP` 使用 `feature_dim=1024`。
- `SigLIP` 使用 `feature_dim=768`。

### 5.1 通用模板

```bash
python src/main.py \
  --name EXP_NAME \
  --FL_algorithm FedMMDP \
  --dataset DATASET \
  --data_root DATA_ROOT \
  --model MODEL_NAME \
  --feature_dim FEATURE_DIM \
  --lr 1e-5 \
  --local_epochs 1 \
  --comm_rounds 20 \
  --batch_size BATCH_SIZE \
  --cluster_method kmeans \
  --n_clusters 5 \
  --cluster_weight 1.0 \
  --rmg_weight 0.0
```

### 5.2 四组核心实验

#### ImageNet + CLIP

```bash
python src/main.py \
  --name fedmmdp_imagenet_clip \
  --FL_algorithm FedMMDP \
  --dataset imagenet \
  --data_root "$IMAGENET_CLIP_ROOT" \
  --model clip \
  --feature_dim 1024 \
  --lr 1e-5 \
  --local_epochs 1 \
  --comm_rounds 20 \
  --batch_size 64 \
  --cluster_method kmeans \
  --n_clusters 5 \
  --cluster_weight 1 \
  --rmg_weight 0
```

#### ImageNet + SigLIP

```bash
python src/main.py \
  --name fedmmdp_imagenet_siglip \
  --FL_algorithm FedMMDP \
  --dataset imagenet \
  --data_root "$IMAGENET_SIGLIP_ROOT" \
  --model siglip \
  --feature_dim 768 \
  --lr 1e-5 \
  --local_epochs 1 \
  --comm_rounds 20 \
  --batch_size 64 \
  --cluster_method kmeans \
  --n_clusters 5 \
  --cluster_weight 1 \
  --rmg_weight 0
```

#### IAPR + CLIP

```bash
python src/main.py \
  --name fedmmdp_iapr_clip \
  --FL_algorithm FedMMDP \
  --dataset iapr \
  --data_root "$IAPR_CLIP_ROOT" \
  --model clip \
  --feature_dim 1024 \
  --lr 1e-5 \
  --local_epochs 1 \
  --comm_rounds 20 \
  --batch_size 64 \
  --cluster_method kmeans \
  --n_clusters 5 \
  --cluster_weight 1 \
  --rmg_weight 0
```

#### IAPR + SigLIP

```bash
python src/main.py \
  --name fedmmdp_iapr_siglip \
  --FL_algorithm FedMMDP \
  --dataset iapr \
  --data_root "$IAPR_SIGLIP_ROOT" \
  --model siglip \
  --feature_dim 768 \
  --lr 1e-5 \
  --local_epochs 1 \
  --comm_rounds 20 \
  --batch_size 64 \
  --cluster_method kmeans \
  --n_clusters 5 \
  --cluster_weight 1 \
  --rmg_weight 0
```

如果你要复现不同聚类策略，可以把 `--cluster_method` 改成：

- `finch`
- `spectral`
- `kmeans`
- `dbscan`

例如：

```bash
python src/main.py \
  --name fedmmdp_imagenet_clip_finch \
  --FL_algorithm FedMMDP \
  --dataset imagenet \
  --data_root "$IMAGENET_CLIP_ROOT" \
  --model clip \
  --feature_dim 1024 \
  --lr 1e-5 \
  --local_epochs 1 \
  --comm_rounds 20 \
  --batch_size 64 \
  --cluster_method finch \
  --partition_level 1 \
  --cluster_weight 1 \
  --rmg_weight 0
```

## 6. Baseline 对比实验

### 6.1 非蒸馏类对比算法

这组算法不依赖额外 public dataset：

- `FedAvg`
- `FedProx`
- `MOON`
- `Harmony`
- `MASA`
- `FedMEMA`
- `RawCLIP`
- `CenterTraining`

先定义一个 bash 函数：

```bash
run_shared_baselines() {
  local prefix=$1
  local dataset=$2
  local data_root=$3
  local model_name=$4
  local feature_dim=$5
  local batch_size=$6

  for alg in FedAvg FedProx MOON Harmony MASA FedMEMA RawCLIP CenterTraining; do
    python src/main.py \
      --name "${prefix}_${alg}" \
      --FL_algorithm "${alg}" \
      --dataset "${dataset}" \
      --data_root "${data_root}" \
      --model "${model_name}" \
      --feature_dim "${feature_dim}" \
      --lr 1e-5 \
      --local_epochs 1 \
      --comm_rounds 20 \
      --batch_size "${batch_size}"
  done
}
```

四组对比实验调用方式：

```bash
rm -rf data_partition/*
run_shared_baselines imagenet_clip imagenet "$IMAGENET_CLIP_ROOT" clip 1024 64

rm -rf data_partition/*
run_shared_baselines imagenet_siglip imagenet "$IMAGENET_SIGLIP_ROOT" siglip 768 64

rm -rf data_partition/*
run_shared_baselines iapr_clip iapr "$IAPR_CLIP_ROOT" clip 1024 64

rm -rf data_partition/*
run_shared_baselines iapr_siglip iapr "$IAPR_SIGLIP_ROOT" siglip 768 64
```

### 6.2 蒸馏类对比算法

这组算法依赖 MSCOCO 2014 public dataset：

- `FedMD`
- `FedDF`
- `Cream`

public MSCOCO 2014 cache 会在数据目录下按模型自动生成，例如 `coco_emb_clip_1024.pt`、`coco_emb_siglip_768.pt`。训练前仍需确认代码中的 MSCOCO 2014 根路径与你的本地环境一致。

```bash
run_public_baselines() {
  local prefix=$1
  local dataset=$2
  local data_root=$3
  local model_name=$4
  local feature_dim=$5
  local batch_size=$6

  for alg in FedMD FedDF Cream; do
    python src/main.py \
      --name "${prefix}_${alg}" \
      --FL_algorithm "${alg}" \
      --dataset "${dataset}" \
      --data_root "${data_root}" \
      --model "${model_name}" \
      --feature_dim "${feature_dim}" \
      --lr 1e-5 \
      --local_epochs 1 \
      --comm_rounds 20 \
      --batch_size "${batch_size}" \
      --pub_data_num 5000
  done
}
```

四组对比实验调用方式：

```bash
rm -rf data_partition/*
run_public_baselines imagenet_clip imagenet "$IMAGENET_CLIP_ROOT" clip 1024 64

rm -rf data_partition/*
run_public_baselines imagenet_siglip imagenet "$IMAGENET_SIGLIP_ROOT" siglip 768 64

rm -rf data_partition/*
run_public_baselines iapr_clip iapr "$IAPR_CLIP_ROOT" clip 1024 64

rm -rf data_partition/*
run_public_baselines iapr_siglip iapr "$IAPR_SIGLIP_ROOT" siglip 768 64
```

## 7. 调试与回归测试

如果你只是想快速检查命令是否能启动，可以把通信轮数调成 2，并关闭 t-SNE：

```bash
python src/main.py \
  --name smoke_fedmmdp_iapr_clip \
  --FL_algorithm FedMMDP \
  --dataset iapr \
  --data_root "$IAPR_CLIP_ROOT" \
  --model clip \
  --feature_dim 1024 \
  --lr 1e-5 \
  --local_epochs 1 \
  --comm_rounds 2 \
  --batch_size 4 \
  --disable_tsne
```

## 8. 目录说明

```text
src/
  main.py                         统一训练入口
  algorithms/                     FedMMDP 与各类 baseline
  datasets/                       数据预处理与联邦数据加载
  networks/                       CLIP / SigLIP / projector 训练脚本
  utils/                          配置、日志、工具函数

preprocessed_imagenet/
preprocessed_iapr/
data_partition/
experiments/
results/
```

## 9. 额外说明

- `SigLIP` 实验需要同时切换三项：`--model siglip`、`SigLIP` 预处理数据目录，以及 `--feature_dim 768`。
- `FedMD`、`FedDF`、`Cream` 除了联邦数据外，还需要单独准备 MSCOCO public dataset。
- 当更换数据集、模态特征目录或者客户端划分配置时，建议重新删除 `data_partition/*`。
- 如果只做训练逻辑调试，建议附加 `--disable_tsne`，避免可视化耗时影响实验排查。
