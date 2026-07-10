# UNet-Semantic-Segmentation-Model-
A PyTorch implementation of UNet for image segmentation tasks.This project includes complete training, validation, and checkpoint management for semantic segmentation.

## Features

 **UNet Architecture** - Classic encoder-decoder architecture with skip connections  
 **Binary Segmentation** - BCEWithLogitsLoss for binary mask prediction  
 **Dice Coefficient Metric** - For evaluating segmentation performance  
 **Checkpoint Management** - Save and resume training with best model tracking  
 **Data Preprocessing** - Automatic train/validation split with image-mask pairing  
 **GPU Support** - CUDA acceleration when available

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- Pillow (PIL)
- tqdm
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd UNet-Semantic-Segmentation-Model
```

2. Install dependencies:
```bash
pip install torch torchvision Pillow tqdm
```

## Project Structure

```
├── m2.py           # Dataset class for loading images and masks
├── model.py        # UNet architecture definition
├── datase.py       # DataLoader configuration
├── train.py        # Training loop script
├── dice.py         # Dice coefficient metric
├── save.py         # Checkpoint saving utility
├── load.py         # Dataset preprocessing and splitting
└── README.md       # This file
```

## Dataset Format

The dataset should be organized as follows:

```
rawdata/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1_mask.jpg
    ├── image2_mask.jpg
    └── ...
```

**Note:** Images and masks must have matching identifiers for proper pairing.

## Usage

### 1. Prepare Your Dataset

Place your raw images and masks in a `rawdata/` folder and run the preprocessing script:

```bash
python load.py
```

This will:
- Match images to their corresponding masks
- Split data into 85% train / 15% validation
- Organize files into the required directory structure under `data/`

### 2. Train the Model

```bash
python train.py
```

The training script will:
- Load the preprocessed data
- Train the UNet model for 10 epochs
- Validate after each epoch
- Save checkpoints for every epoch
- Save the best model based on validation Dice score

### 3. Resume Training (Optional)

Modify `train.py` to load a checkpoint:

```python
checkpoint = torch.load("checkpoint_5.pt")
model.load_state_dict(checkpoint["modelstate"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"] + 1
```

## Model Architecture

**UNet** - Encoder-Decoder with Skip Connections:

```
Input (3 channels)
    ↓
Down 1: 64 → MaxPool
    ↓
Down 2: 128 → MaxPool
    ↓
Down 3: 256 → MaxPool
    ↓
Down 4: 512 → MaxPool
    ↓
Bottleneck: 1024 (Double Conv)
    ↓
Up 1: 512 (+ skip from Down 4)
    ↓
Up 2: 256 (+ skip from Down 3)
    ↓
Up 3: 128 (+ skip from Down 2)
    ↓
Up 4: 64 (+ skip from Down 1)
    ↓
Output: 1 channel (segmentation mask)
```

## Configuration

Edit parameters in the respective scripts:

**`datase.py`** - Batch size and data loading:
```python
train_loader = DataLoader(train_data, batch_size=4, num_workers=0, shuffle=True)
```

**`train.py`** - Training hyperparameters:
```python
num_epochs = 10
lr = 1e-4  # Learning rate (in optimizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**`load.py`** - Train/validation split:
```python
SPLIT_RATIO = 0.85  # 85% train, 15% validation
IMG_SIZE = 256      # Resize images to 256x256
```

## Loss Functions & Metrics

- **Loss**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- **Metric**: Dice Coefficient (F1 score) for segmentation quality

## Outputs

During training, the script generates:
- `checkpoint_0.pt`, `checkpoint_1.pt`, ... - Model checkpoints for each epoch
- `best.pt` - Best model based on validation Dice score

## Example Training Loop

```bash
$ python load.py
# Output: Successfully linked 100 image-mask matches
# Output: Splitting into 85 train and 15 validation items

$ python train.py
# Output: Epoch 1/10 [Train] ...
# Output: Epoch 1, Loss 0.4523
# Output: Validation Loss: 0.3421
# ... (continues for 10 epochs)
```

## Tips for Better Performance

1. **Data Augmentation** - Modify transforms in `m2.py` to add rotations, flips, etc.
2. **Batch Size** - Increase batch size if GPU memory allows (4 → 8 or 16)
3. **Learning Rate** - Adjust in `train.py` (currently 1e-4)
4. **Epochs** - Increase `num_epochs` in `train.py` for better convergence
5. **Image Resolution** - Change `img_size` parameter (currently 256)

