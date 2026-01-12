````md
# Real-Time Face Emotion Recognition using Deep Learning

A production-ready **real-time facial emotion recognition system** built using **TensorFlow, EfficientNet, and OpenCV**, with a strong emphasis on **deep learning model development, GPU-accelerated training, and real-time deployment with confidence scores**.

This project demonstrates a complete **end-to-end AI workflow**: from **GPU-based mixed-precision training** to real-time inference on live webcam video.

---

## 📌 Project Highlights

- Custom-trained CNN-based emotion recognition model
- **100% GPU-trained** using mixed precision
- Two-phase **transfer learning + fine-tuning** strategy
- EfficientNet backbone for high accuracy with low inference cost
- Real-time face detection using OpenCV
- Emotion predictions with **confidence percentages**
- Decoupled **client–server inference architecture**
- Optimized for real-time CPU inference

---

## 🎯 Emotion Classes

The model predicts the following facial emotions:

```text
angry | disgust | fear | happy | neutral | sad | surprise
````

---

## 🧠 Model Development (Primary Focus)

### Dataset & Preprocessing

* Facial images organized by emotion class
* Converted to **grayscale** to reduce noise and computational cost
* Resized to **224 × 224**
* Pixel normalization to `[0, 1]`
* Directory-based training and validation generators

---

### Data Augmentation Strategy

To ensure strong generalization to **real-world webcam conditions**, aggressive augmentation was applied during GPU training:

* Rotation (±30°)
* Width & height shifts (±20%)
* Zoom and shear transformations
* Brightness variation
* Horizontal flipping

This makes the model robust to:

* Head pose changes
* Illumination variation
* Camera noise and motion blur

---

## 🏗️ Model Architecture

The model is built around a **pretrained EfficientNetB0 backbone**, combined with a custom classification head.

### Key Engineering Decisions

* EfficientNet provides an excellent **accuracy-to-efficiency ratio**
* Grayscale inputs mapped to 3 channels via a `Conv2D` layer
* Global Average Pooling reduces overfitting
* Softmax output enables **probabilistic confidence scoring**

### Architecture Overview

```text
Grayscale Input (224×224×1)
        ↓
Conv2D (1 → 3 channels)
        ↓
EfficientNetB0 (ImageNet)
        ↓
Global Average Pooling
        ↓
Dropout (0.3)
        ↓
Dense (Softmax)
```

---

## ⚙️ GPU-Based Mixed Precision Training

**All model training was performed on GPU** using **mixed precision**, providing:

* Faster training throughput
* Lower GPU memory usage
* Larger batch size (`64`)
* Improved numerical efficiency

GPU availability was explicitly detected and configured before training.

---

## 🔁 Two-Phase Training Strategy

### Phase 1 — Feature Extraction (GPU)

* EfficientNet backbone **frozen**
* Only classification head trained
* Learning rate: `1e-3`
* Allows stable emotion-specific feature learning

### Phase 2 — Fine-Tuning (GPU)

* Last 30 layers of EfficientNet **unfrozen**
* Learning rate reduced to `1e-5`
* Adapts pretrained ImageNet features to facial emotion domain
* Improves accuracy and confidence calibration

---

### Training Controls & Callbacks

| Callback          | Purpose                    |
| ----------------- | -------------------------- |
| ModelCheckpoint   | Save best validation model |
| EarlyStopping     | Prevent overfitting        |
| ReduceLROnPlateau | Adaptive learning rate     |

Final trained artifact:

```text
best_model2.keras
```

---

## 📊 Confidence-Based Predictions

The model outputs a probability distribution across emotion classes:

```python
emotion = argmax(prediction)
confidence = max(prediction) × 100
```

This enables:

* Interpretable predictions
* Stable real-time visualization
* Reduced prediction jitter across frames

---

## 🎥 Real-Time Emotion Recognition Pipeline

### Face Detection

* Haar Cascade classifier (OpenCV)
* Optimized for real-time CPU execution

### Preprocessing Consistency

Inference preprocessing **exactly matches training**:

* Grayscale conversion
* Resize to `224 × 224`
* Normalization
* Shape `(1, 224, 224, 1)`

### Visualization

* Bounding box drawn around detected face
* Emotion label with confidence percentage
* Frame-skipping optimization for smooth FPS

---

## 🧩 Deployment Architecture

The system uses a **client–server architecture**:

### Client (Windows)

* Webcam capture
* Face detection & visualization
* Sends preprocessed face data via TCP sockets

### Server (WSL / Linux)

* Loads trained TensorFlow model once
* Performs inference
* Returns emotion index & confidence score

Communication stack:

* TCP sockets
* JSON
* Base64 encoding
* Newline-delimited messages

---

## 🔁 End-to-End Flow

```text
GPU Model Training → Saved Model → Real-Time Inference → Emotion + Confidence
```

---

## ⚡ Performance Optimizations

* Grayscale inference
* Frame skipping (process every N frames)
* Single-face inference per frame
* Model loaded once in memory
* Socket timeouts to avoid UI blocking

---

## 🛠️ Tech Stack

### Machine Learning

* TensorFlow
* Keras
* EfficientNet
* NumPy

### Computer Vision

* OpenCV

### Systems & Networking

* TCP Sockets
* JSON
* Base64 Encoding

### Hardware & Platform

* **GPU (Training)**
* Windows (Client)
* WSL / Linux (Inference Server)

---

## ▶️ Running the Project

### Start Inference Server (WSL)

```bash
python3 wsl_server.py
```

### Run Real-Time Client (Windows)

```bash
python recog.py
```

---



---

