# Federated Multimodal Domain Personalization (FedMMDP)

## Overview

FedMMDP is a federated learning framework designed for multimodal domain personalization. It supports heterogeneous clients with different modalities (e.g., image, text) and leverages privacy-preserving techniques such as homomorphic encryption. The system is built to enable domain-specific personalization while maintaining data privacy and efficient multimodal alignment.

## Features

- **Federated Multimodal Learning:** Supports clients with different data modalities.
- **Domain Personalization:** Enables personalized models for each client/domain.
- **Privacy Protection:** Integrates homomorphic encryption for secure aggregation.
- **Flexible Model Backbone:** Uses CLIP as the server model, with support for custom projectors.
- **Visualization Tools:** Provides tools for feature and hierarchy visualization.
- **Dataset Partitioning:** Supports heterogeneous and homogeneous data partitioning.

## Environment Setup

1. **Clone the repository**

   ```sh
   git clone https://github.com/Toony254/FedMMDP.git
   cd FedMMDP
   ```
2. **Install dependencies**

   ```sh
   conda create -n FedMMDP python=3.10
   conda activate FedMMDP
   pip install -r requirements.txt
   ```
3. **Download datasets**

   ```sh
   nohup huggingface-cli download --repo-type dataset --resume-download gmongaras/Imagenet21K_Recaption --local-dir ~/data/zs/FedMMDP/dataset >outputs/download_dataset.log 2>&1 &
   nohup ./hfd.sh gmongaras/Imagenet21K_Recaption --dataset --local-dir ~/data/zs/FedMMDP/dataset >outputs/download_dataset.log 2>&1 &
   nohup find dataset/data -name "*.parquet" -exec parquet-tools inspect {} \; >outputs/check_parquet.log 2>&1 &
   ```

## Usage

### Train the projector module

```sh
python train_projector.py --projector bottleneck >outputs/train_projector.log 2>&1 &
```

### Split dataset by domain

```sh
python src/datasets/split_domain_dataset.py >outputs/preprocess_datasets.log 2>&1 &
```


### Train FedMMDP with CLIP as server model

```sh
# CUDA_VISIBLE_DEVICES=3 python src/main.py --name RawCLIP --FL_algorithm RawCLIP --local_epochs 1 --comm_rounds 1 --batch_size 64 --model clip >outputs/output_rawclip.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python src/main.py --name CenterTraining --FL_algorithm CenterTraining --local_epochs 1 --comm_rounds 30 --batch_size 64 --model clip >outputs/output_center_clip.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python src/main.py --name CenterTraining --FL_algorithm CenterTraining --local_epochs 1 --comm_rounds 30 --batch_size 64 --model resnet >outputs/output_center_resnet.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python src/main.py --name FedMMDP-moon --FL_algorithm MOON --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 64 >outputs/output_moon.log 2>&1 &
Non-IID
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-avg --FL_algorithm FedAvg --lr 1e-5 --local_epochs 1 --comm_rounds 20  --batch_size 64 --model clip >outputs/output_avg_clip.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-prox --FL_algorithm FedProx --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 64 >outputs/output_prox.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-md --FL_algorithm FedMD --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 64 --pub_data_num 5000 >outputs/output_md.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --name FedMMDP-df --FL_algorithm FedDF --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 64 --pub_data_num 5000 >outputs/output_df.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python src/main.py --name FedMMDP-Cream --FL_algorithm Cream --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 64 --pub_data_num 5000 >outputs/output_cream.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --name FedMMDP-Harmony --FL_algorithm Harmony --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 64 >outputs/output_Harmony.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --name FedMMDP-MASA --FL_algorithm MASA --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 64 >outputs/output_MASA.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-MEMA --FL_algorithm FedMEMA --lr 1e-5 --local_epochs 1 --comm_rounds 20 --model clip --batch_size 64 >outputs/output_FedMEMA.log 2>&1 &

IID
CUDA_VISIBLE_DEVICES=2 python src/main.py --name FedMMDP-avg --FL_algorithm FedAvg --lr 1e-5 --local_epochs 1 --alpha 5.0 --comm_rounds 20  --batch_size 64 --model clip >outputs/output_avg_clip.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-prox --FL_algorithm FedProx --lr 1e-5 --local_epochs 1 --alpha 5.0 --comm_rounds 20 --model clip --batch_size 64 >outputs/output_prox.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --name FedMMDP-md --FL_algorithm FedMD --lr 1e-5 --local_epochs 1 --alpha 5.0 --comm_rounds 20 --model clip --batch_size 64 --pub_data_num 5000 >outputs/output_md.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --name FedMMDP-df --FL_algorithm FedDF --lr 1e-5 --local_epochs 1 --alpha 5.0 --comm_rounds 20 --model clip --batch_size 64 --pub_data_num 5000 >outputs/output_df.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python src/main.py --name FedMMDP-Cream --FL_algorithm Cream --lr 1e-5 --local_epochs 1 --alpha 5.0 --comm_rounds 20 --model clip --batch_size 64 --pub_data_num 5000 >outputs/output_cream.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --name FedMMDP-Harmony --FL_algorithm Harmony --lr 1e-5 --local_epochs 1 --alpha 5.0 --comm_rounds 20 --model clip --batch_size 64 >outputs/output_Harmony.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python src/main.py --name FedMMDP-MASA --FL_algorithm MASA --lr 1e-5 --local_epochs 1 --alpha 5.0 --comm_rounds 20 --model clip --batch_size 64 >outputs/output_MASA.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --name FedMMDP-FedMEMA --FL_algorithm FedMEMA --lr 1e-5 --local_epochs 1 --alpha 5.0 --comm_rounds 20 --model clip --batch_size 64 >outputs/output_FedMEMA.log 2>&1 &
```

## Project Structure

```
├── src/
│   ├── main.py                # Entry point for federated training
│   ├── algorithms/            # Federated learning algorithms and trainers
│   ├── criterions/            # Loss functions
│   ├── datasets/              # Dataset loading, partitioning, and preprocessing
│   ├── losses/                # Custom loss modules
│   ├── networks/              # Model architectures and projector modules
│   ├── utils/                 # Utility functions and config parsers
│   ├── visualization/         # Visualization scripts (t-SNE, hierarchy, etc.)
├── data/                      # Datasets (filtered, full, partitioned)
├── saved/                     # Saved models and checkpoints
├── saved_clients/             # Client-specific model checkpoints
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
```

## Main Components

- **src/main.py**Launches the federated training process with configurable parameters.
- **src/algorithms/**Contains the core federated learning logic, including multimodal trainers and optimizers.
- **src/networks/train_projector.py**Trains the projector module for feature alignment between modalities.
- **src/datasets/**Handles dataset loading, preprocessing, and partitioning for federated scenarios.
- **src/visualization/**Scripts for visualizing feature distributions (e.g., t-SNE), client data, and label hierarchies.
- **src/utils/**
  Helper functions for configuration, logging, and data processing.

## Visualization

- **Feature Visualization:**Use scripts in `src/visualization/` to visualize learned representations with t-SNE and other methods.
- **Hierarchy Visualization:**
  Generate interactive HTML visualizations of the label hierarchy using `src/datasets/imagenet_label.py`.

## Notes

- Make sure to adjust dataset paths in scripts as needed.
- For large-scale experiments, use `nohup` and redirect logs to `output/` for tracking.
- The system supports both homogeneous and heterogeneous data partitioning; configure via YAML or command-line arguments.

## Citation

If you use this codebase, please cite the corresponding paper.

---

```

```
