<div align="center">
  <h1>🚀 Real-Time Facial Emotion Detection System</h1>
  <p><strong>Deep Learning | Computer Vision | Transfer Learning | Real-Time AI Deployment</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
  [![Keras](https://img.shields.io/badge/Keras-API-red.svg)](https://keras.io/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff69b4.svg)](https://streamlit.io/)
</div>

<hr />

## 📌 Overview

Developed a real-time facial emotion recognition system using **Deep Learning** and **Transfer Learning**.  
The system detects and classifies human emotions from images and live webcam streams with an interactive analytics dashboard.

This project demonstrates an end-to-end Machine Learning system design:
- Data Preprocessing
- Transfer Learning (MobileNetV2)
- Model Training
- Real-time Inference
- Web App Deployment via Streamlit

<hr />

## 🧠 Problem Statement

Emotion recognition from facial expressions is a complex multi-class classification problem with subtle inter-class differences and high intra-class variation. 
This system leverages CNN-based feature extraction with **MobileNetV2** to improve generalization and inference efficiency.

<hr />

## 🏗 Architecture

```text
Input (Image/Webcam)  
      ↓  
Face Detection (OpenCV Haar Cascade)  
      ↓  
Preprocessing (Resize 48x48, Normalize, RGB Conversion)  
      ↓  
MobileNetV2 (Transfer Learning)  
      ↓  
Dense Classification Head  
      ↓  
Softmax Output (7 Emotions)  
```

<hr />

## 📊 Model Details

| Component | Implementation |
|-----------|----------------|
| **Base Model** | MobileNetV2 (ImageNet Weights) |
| **Input Size** | 48x48x3 |
| **Output Classes** | 7 Emotions |
| **Optimizer** | Adam |
| **Loss** | Categorical Crossentropy |
| **Framework** | TensorFlow / Keras |
| **Deployment** | Streamlit |

<hr />

## 🎯 Emotions Detected

The system classifies facial expressions into **7 distinct emotions**:
- 😠 **Angry**
- 🤢 **Disgust**
- 😨 **Fear**
- 😄 **Happy**
- 😢 **Sad**
- 😲 **Surprise**
- 😐 **Neutral**

<hr />

## 📈 Performance

- **Training Accuracy:** ~65%
- **Validation Accuracy:** ~60%
- **Real-Time Inference:** Supported
- **Optimization:** RGB pipeline optimized for MobileNet compatibility

<hr />

## 💡 Key Technical Highlights

- ✔️ **Implemented Transfer Learning** (MobileNetV2)  
- ✔️ **Designed custom classification head**  
- ✔️ **Built real-time webcam inference system**  
- ✔️ **Solved RGB vs grayscale model compatibility issue**  
- ✔️ **Developed probability confidence visualization dashboard**  
- ✔️ **Structured production-ready ML project**  

<hr />

## 📂 Project Structure

```bash
emotion-detection/  
│  
├── src/  
│   ├── train.py          # Model training script
│   └── realtime.py       # Live webcam inference script
│  
├── dataset/              # Training & Validation data
├── models/               # Saved trained models
├── app.py                # Streamlit dashboard application
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

<hr />

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/VedantVH/Real-Time-Facial-Emotion-Detection-System-Using-Deep-Learning.git
cd Real-Time-Facial-Emotion-Detection-System-Using-Deep-Learning/emotion-detection
```

### 2️⃣ Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Train the Model (Optional)
```bash
cd src
python train.py
```

### 4️⃣ Run Real-Time Webcam Detection
```bash
python src/realtime.py
```

### 5️⃣ Launch the Web App
```bash
streamlit run app.py
```

<hr />

## 🧪 Future Improvements

- [ ] Fine-tuning entire MobileNet layers  
- [ ] Model quantization for edge devices  
- [ ] Multi-face tracking  
- [ ] Cloud deployment (AWS / Streamlit Cloud)  
- [ ] Emotion analytics tracking over time  

<hr />

<div align="center">
  <p>Developed with ❤️ by Vedant.</p>
</div>
