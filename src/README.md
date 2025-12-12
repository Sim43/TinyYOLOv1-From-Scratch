# TinyYOLOv1 From Scratch - Source Code Guide

This directory contains a complete implementation of TinyYOLOv1 for person detection, built from scratch using PyTorch. The codebase follows a modular design, making it easy to understand and extend.

## 🎯 Overview

This project implements a simplified YOLOv1-style object detector specifically trained for detecting pedestrians. The model uses a 7×7 grid to predict bounding boxes and class probabilities, making it lightweight and suitable for learning the fundamentals of object detection.

## 📁 File Structure & Purpose

### Core Configuration
**`config.py`** ⚙️  
Central configuration file that defines all hyperparameters and paths:
- Grid size (S=7), number of classes (C=1 for person detection)
- Training parameters (epochs, batch size, learning rate)
- Loss function weights (coordinate loss, no-object loss)
- Data and output directory paths

### Data Handling
**`dataset.py`** 📦  
Custom PyTorch Dataset class for the PennFudanPed dataset:
- Loads images and corresponding mask files
- Converts instance masks to bounding boxes
- Splits data into train/validation sets (80/20 split)
- Transforms bounding boxes into YOLO format (normalized center coordinates, width, height)
- Builds target tensors of shape `(S, S, 5+C)` for training

**Key methods:**
- `_mask_to_boxes()`: Extracts bounding boxes from segmentation masks
- `__getitem__()`: Returns preprocessed image and target tensor

### Model Architecture
**`model.py`** 🧠  
Defines the TinyYOLOv1 neural network:
- **Backbone**: Lightweight CNN with 5 convolutional layers + batch normalization
- **Head**: Final convolution layer producing `(S, S, 5+C)` output
  - Each cell predicts: `[tx, ty, tw, th, obj_score, class_prob]`
  - `tx, ty`: Cell-relative coordinates (0-1 within cell)
  - `tw, th`: Image-relative width/height (0-1 across full image)
  - `obj_score`: Objectness confidence
  - `class_prob`: Class probability (person in this case)

### Loss Function
**`loss.py`** 📉  
Implements YOLOv1-style loss function:
- **Coordinate loss**: MSE on bounding box coordinates (weighted by `lambda_coord`)
- **Objectness loss**: Binary cross-entropy for cells with/without objects
- **No-object loss**: Penalizes false positives (weighted by `lambda_noobj`)
- **Class loss**: Binary cross-entropy for class prediction
- Uses square root on width/height for training stability

### Training Script
**`train.py`** 🚀  
Main training loop that orchestrates the entire training process:
1. Loads configuration and initializes datasets
2. Creates model, loss function, and optimizer (AdamW)
3. Trains for specified epochs with progress bars
4. Validates after each epoch
5. Saves best checkpoint based on validation loss
6. Includes gradient clipping for stability

### Utility Functions
**`utils.py`** 🔧  
Helper functions for target building and inference:
- **`build_yolo_target()`**: Converts ground truth boxes to YOLO grid format
  - Assigns objects to grid cells based on center coordinates
  - Handles multiple objects per cell (keeps largest)
- **`decode_predictions()`**: Converts model output to bounding boxes
  - Applies sigmoid to get probabilities
  - Filters by confidence threshold
  - Converts cell-relative to image-relative coordinates
- **`nms_normalized()`**: Non-maximum suppression to remove duplicate detections

### Inference Scripts
**`infer_image.py`** 🖼️  
Runs inference on a single image:
- Loads trained checkpoint
- Processes image through model
- Draws bounding boxes with confidence scores
- Displays result in OpenCV window

**`infer_video.py`** 🎥  
Real-time inference on video files or webcam:
- Supports video files or webcam (pass `0` for default camera)
- Processes frames sequentially
- Press 'q' to quit

## 🔄 Complete Workflow

### 1. Data Preparation
The dataset (`PennFudanPed`) should be placed in `data/PennFudanPed/` with:
- `PNGImages/`: Original images
- `PedMasks/`: Segmentation masks (each pedestrian has unique pixel value)

### 2. Training Process
```
config.py → dataset.py → model.py → loss.py → train.py
```

1. **Config** defines all parameters
2. **Dataset** loads and preprocesses images/masks into YOLO format
3. **Model** processes images through CNN
4. **Loss** computes error between predictions and targets
5. **Train** optimizes model weights using backpropagation

### 3. Inference Pipeline
```
Checkpoint → Model → Decode → NMS → Draw Boxes
```

1. Load saved checkpoint with model weights
2. Forward pass through model
3. Decode predictions to bounding boxes
4. Apply NMS to remove duplicates
5. Draw boxes on original image

## 🎓 Key Concepts

**YOLO Grid System**: The image is divided into a 7×7 grid. Each cell is responsible for detecting objects whose center falls within that cell.

**Target Format**: `(S, S, 5+C)` tensor where:
- First 4 values: `[tx, ty, tw, th]` - bounding box parameters
- 5th value: `obj` - object presence (1 or 0)
- Remaining C values: one-hot class encoding

**Training Strategy**: The model learns to:
- Predict accurate bounding box coordinates
- Distinguish between cells with/without objects
- Classify detected objects (person vs background)

---

*This implementation prioritizes clarity and educational value while maintaining a functional object detection system.*

