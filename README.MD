# DSPIGCN: Dual-stream Physics-informed Graph Convolutional Network for Reliable Pedestrian Trajectory Prediction

## Introduction
DSPIGCN is a deep learning model for pedestrian trajectory prediction that incorporates physics-informed constraints.

## Setup

### Prerequisites
- Python 3.8
- PyTorch
- CUDA (if using GPU)

### Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
The project expects the datasets to be located in the `datasets/` directory. The supported datasets include:
- eth
- hotel
- univ
- zara1
- zara2
- trajnet
- sdd

Ensure your data is organized as follows:
```
datasets/
    eth/
        test/
        train/
        val/
    ...
```

## Training
To train the model, use `train.py`. You can specify various hyperparameters and the target dataset.

### Common Arguments
- `--dataset`: Dataset to train on (e.g., `eth`, `hotel`, `univ`, `zara1`, `zara2`).
- `--tag`: A unique tag for the model checkpoint directory.
- `--num_epochs`: Number of training epochs. Default: 200.
- `--batch_size`: Batch size. Default: 128.
- `--lr`: Learning rate. Default: 0.001.
- `--use_lrschd`: Enable learning rate scheduler.
- `--n_stgcnn`: Number of ST-GCNN layers. Default: 1.
- `--n_txpcnn`: Number of TXPCNN layers. Default: 5.
- `--obs_seq_len`: Observation sequence length. Default: 8.
- `--pred_seq_len`: Prediction sequence length. Default: 12.

### Example Commands
You can refer to `train.sh` for batch training commands. Here are examples for individual datasets:

**Train on ETH:**
```bash
python train.py --lr 0.001 --n_stgcnn 1 --n_txpcnn 5 --dataset eth --tag social-stgcnn-eth --use_lrschd --num_epochs 200
```

Checkpoints will be saved in `KDD_checkpoint/<tag>/`.

## Evaluation
To evaluate the trained models, use `test.py`.

```bash
python test.py
```

### Results
Evaluation results (ADE, FDE, etc.) will be saved in the `Results/KDD/` directory.

