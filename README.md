# Road-Segmentation-Aerial-Images
Road Segmentation using Deep CNNs (U-Net, Seg-Net ,Attention U-Net) on Aerial Images

A comparative study on U-Net, SegNet, and Attention U-Net for binary segmentation of road networks from aerial imagery using the **Massachusetts Roads Dataset**. This project demonstrates how attention-based CNN models can enhance road extraction in real-world conditions such as smart cities, autonomous vehicles, and disaster response systems.

---

##  Overview

Accurate segmentation of road networks from aerial images is crucial in applications such as autonomous navigation, urban planning, and emergency response. This project uses **Convolutional Neural Networks (CNNs)** for semantic segmentation and compares the performance of:
- U-Net
- SegNet
- Attention U-Net

All models were trained and evaluated on the Massachusetts Roads Dataset with consistent preprocessing, loss functions, and metrics.

---

## 📂 Dataset

- **Massachusetts Roads Dataset** (Kaggle)
- 1171 high-resolution aerial images (1500x1500 px) with binary road masks
- Preprocessed into 256x256 patches for training

### 📁 Directory Structure
├── train/
├── train_labels/
├── val/
├── val_labels/
├── test/
└── test_labels/

---

##  Models

### 1. U-Net
Encoder-decoder architecture with skip connections for spatial context preservation.

### 2. SegNet
Memory-efficient encoder-decoder model using max-pooling indices for non-linear upsampling.

### 3. Attention U-Net
Enhanced U-Net with attention gates to focus on salient spatial features and suppress background noise.

---

## Preprocessing

- Resizing: All images and masks resized to `256x256`
- Normalization: Image values scaled to [0, 1]
- Binarization: Masks thresholded to 0 (background) or 1 (road)
- Converted into NumPy arrays for training

---

## Training Configuration

| Parameter              | Value                        |
|------------------------|------------------------------|
| Optimizer              | Adam                         |
| Learning Rate          | 0.0001 (Reduced on Plateau)  |
| Batch Size             | 16                           |
| Epochs                 | Up to 100 (Early Stopping)   |
| Loss Function          | BCE + Dice Loss              |
| Activation             | Sigmoid                      |
| Split                  | 80% Train / 20% Validation   |
| Callbacks              | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Dice Coefficient
- Intersection over Union (IoU)

### 🧾 Final Results:

| Model           | Accuracy | Precision | Recall | F1 Score | IoU   | Dice  |
|------------------|----------|-----------|--------|----------|-------|--------|
| **U-Net**         | 0.966    | 0.671     | 0.608  | 0.638    | 0.468 | 0.638 |
| **Attention U-Net**| **0.967** | **0.677** | **0.617** | **0.645** | **0.477** | **0.645** |
| **SegNet**        | 0.962    | 0.633     | 0.534  | 0.580    | 0.408 | 0.580 |

 **Attention U-Net** outperformed others due to its attention mechanism which improves segmentation of fine features and suppresses irrelevant background.

---

## Visual Results

| Input | Ground Truth | U-Net | Attention U-Net | SegNet |
|-------|---------------|-------|------------------|--------|


---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/road-segmentation-aerial-images.git
cd road-segmentation-aerial-images

# Install requirements
pip install -r requirements.txt

# Run training
python train.py

# Evaluate results
python evaluate.py
