# F0 Estimation using Deep Salience Representation

A deep learning approach for fundamental frequency (F0) estimation from audio using Harmonic Constant-Q Transform (HCQT) and Convolutional Neural Networks (CNNs). This project implements a fully convolutional network that learns to predict pitch salience representations from multi-harmonic time-frequency representations.

## Overview

This project implements a CNN-based F0 estimation system that:

1. Computes Harmonic Constant-Q Transform (HCQT) from audio signals
2. Trains a deep CNN to predict pitch salience maps
3. Extracts F0 contours from predicted salience representations
4. Evaluates performance using standard melody evaluation metrics

The model architecture follows a fully convolutional design with 5 convolutional layers, processing 6 harmonic channels to produce a single-channel salience representation.

## Features

- **HCQT-based preprocessing**: Multi-harmonic time-frequency representation capturing harmonic relationships
- **Deep CNN architecture**: Fully convolutional network with batch normalization
- **End-to-end training**: Direct mapping from audio to salience maps
- **Comprehensive evaluation**: Uses mir_eval metrics for F0 estimation evaluation
- **Sliding window inference**: Handles variable-length audio tracks efficiently

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone the repository:

```bash
git clone https://github.com/ishaanjagyasi/F0_Detection_deep_salience_representation_CNN.git
cd F0_Detection_deep_salience_representation_CNN
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

The training and evaluation datasets are available on Google Drive:

**[Download Data from Google Drive](https://drive.google.com/drive/folders/1sR98wQLOWZbab_LHCOdWzLi8-lqFRNxp?usp=sharing)**

After downloading, extract the data to match the following directory structure:

```
Data/
├── Training_Validation/
│   └── MedleyDB-Pitch/
│       ├── audio/          # Audio files (.wav)
│       ├── pitch/          # F0 annotations (.csv)
│       └── medleydb_pitch_metadata.json
└── Evaluation/
    └── vocadito/
        ├── Audio/          # Evaluation audio files
        └── Annotations/
            └── F0/         # Ground truth F0 annotations
```

**Important**: The `Data/Evaluation/` and `Data/Training_Validation/` folders are excluded from git (see `.gitignore`) due to their large size.

## Usage

### 1. Data Preparation

First, prepare the training data by computing HCQT representations and creating target salience maps:

```bash
python data_set_prep.py
```

This script will:

- Compute HCQT for all training audio files
- Generate target salience representations from F0 annotations
- Create train/validation splits (85/15)
- Save processed data for efficient training

### 2. Training

Train the deep salience CNN model:

```bash
python train_model.py
```

Training parameters can be adjusted in `train_model.py`:

- `LEARNING_RATE`: Learning rate (default: 0.001)
- `BATCH_SIZE`: Batch size (default: 1 for variable-length tracks)
- `NUM_EPOCHS`: Number of training epochs (default: 50)
- `PATIENCE`: Early stopping patience (default: 10)

The model will be saved to `models/final_model.pt` and checkpoints will be saved periodically.

### 3. Evaluation

Evaluate the trained model on the evaluation dataset:

```bash
python evaluate.py
```

This will:

- Load the trained model
- Process evaluation audio files
- Extract F0 contours from predicted salience maps
- Compute evaluation metrics using mir_eval
- Save results to `evaluation_results/evaluation_results.json`

## Project Structure

```
.
├── data_set_prep.py          # Data preprocessing and HCQT computation
├── train_model.py            # Model training script
├── evaluate.py               # Model evaluation script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── Data/
│   ├── Training_Validation/  # Training data (excluded from git)
│   ├── Evaluation/           # Evaluation data (excluded from git)
│   └── processed_data/       # Preprocessed data cache
│
├── models/
│   └── final_model.pt       # Trained model weights
│
├── results/
│   ├── training_curves.png   # Training loss curves
│   └── training_history.json # Training history
│
└── evaluation_results/
    └── evaluation_results.json  # Evaluation metrics
```

## Model Architecture

The Deep Salience CNN follows a fully convolutional architecture:

- **Input**: HCQT representation `(batch, 6, 360, T)` - 6 harmonic channels
- **Layer 1**: 128 filters, 5×5 kernel
- **Layer 2**: 64 filters, 5×5 kernel
- **Layer 3**: 64 filters, 3×3 kernel
- **Layer 4**: 64 filters, 3×3 kernel
- **Layer 5**: 8 filters, 70×3 kernel (captures octave relationships)
- **Output**: 1 filter, 1×1 kernel with sigmoid activation → `(batch, 1, 360, T)`

All layers use:

- Batch normalization
- ReLU activation (except final layer uses sigmoid)
- Zero padding to maintain spatial dimensions

## Configuration

Key parameters can be adjusted in the respective configuration classes:

### Data Configuration (`data_set_prep.py`)

- `SAMPLE_RATE`: 22050 Hz
- `HOP_LENGTH`: 256 samples (~11.6ms)
- `N_BINS`: 360 frequency bins
- `BINS_PER_OCTAVE`: 60 (20 cents per bin)
- `HARMONICS`: [0.5, 1, 2, 3, 4, 5]

### Training Configuration (`train_model.py`)

- `LEARNING_RATE`: 0.001
- `BATCH_SIZE`: 1
- `NUM_EPOCHS`: 50
- `PATIENCE`: 10 (early stopping)

### Evaluation Configuration (`evaluate.py`)

- `SALIENCE_THRESHOLD`: 0.1
- `MIN_F0`: 80 Hz
- `MAX_F0`: 1000 Hz
- `MEDIAN_FILTER_SIZE`: 5 frames
- `GAUSSIAN_SIGMA`: 1.0 frames

## Results

Evaluation metrics are computed using `mir_eval` and include:

- Overall Accuracy (OA)
- Raw Pitch Accuracy (RPA)
- Raw Chroma Accuracy (RCA)
- Voicing Recall (VR)
- Voicing False Alarm (VFA)

Results are saved to `evaluation_results/evaluation_results.json` with both aggregated statistics and per-track scores.

## Citation

If you use this code, please cite the original paper on deep salience representation for F0 estimation.

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or issues, please open an issue on the GitHub repository.
