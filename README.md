This project implements a **real-time face emotion recognition system** using **OpenCV** for face detection, **TensorFlow/Keras** for emotion classification, and **socket-based client–server communication** between Windows and WSL.
The system captures live webcam video, detects faces, preprocesses them, sends them to a TensorFlow model running on a WSL server, and displays predicted emotions with confidence scores in real time.

## 🚀 Key Features

- Real-time webcam-based emotion recognition
- Haar Cascade face detection
- TensorFlow deep learning model inference
- Lightweight TCP socket communication
- Windows client + WSL server architecture
- JSON + Base64 data transfer (safe & reliable)
- FPS optimization by processing every Nth frame

---

## 🧠 System Architecture Overview

- **Client (Windows)**  
  - Captures webcam frames
  - Detects faces using OpenCV
  - Preprocesses face images
  - Sends face data to server
  - Displays emotion predictions

- **Server (WSL / Linux)**  
  - Loads trained TensorFlow model
  - Receives face data
  - Performs emotion prediction
  - Sends back emotion index & confidence

---
