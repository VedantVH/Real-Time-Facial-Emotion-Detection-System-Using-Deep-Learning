# 🚀 Real-Time Facial Emotion Detection System  
### Deep Learning | Computer Vision | Transfer Learning | Real-Time AI Deployment  

-----------------------------------

## 📌 Overview

Developed a real-time facial emotion recognition system using Deep Learning and Transfer Learning.  
The system detects and classifies human emotions from images and live webcam streams with an interactive analytics dashboard.

This project demonstrates end-to-end ML system design:
- Data preprocessing
- Transfer learning
- Model training
- Real-time inference
- Deployment via Streamlit

-----------------------------------

## 🧠 Problem Statement

Emotion recognition from facial expressions is a multi-class classification problem with subtle inter-class differences and high intra-class variation.

This system leverages CNN-based feature extraction with MobileNetV2 to improve generalization and inference efficiency.

-----------------------------------

## 🏗 Architecture

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

-----------------------------------
## 📊 Model Details

| Component | Implementation |
|------------|----------------|
| Base Model | MobileNetV2 (ImageNet Weights) |
| Input Size | 48x48x3 |
| Output Classes | 7 Emotions |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Framework | TensorFlow / Keras |
| Deployment | Streamlit |

-----------------------------------

## 🎯 Emotions Detected

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

-----------------------------------

## 📈 Performance

- Training Accuracy: ~65%
- Validation Accuracy: ~60%
- Real-time inference supported
- Optimized RGB pipeline for MobileNet compatibility

-----------------------------------

## 💡 Key Technical Highlights

✔ Implemented Transfer Learning (MobileNetV2)  
✔ Designed custom classification head  
✔ Built real-time webcam inference system  
✔ Solved RGB vs grayscale model compatibility issue  
✔ Developed probability confidence visualization dashboard  
✔ Structured production-ready ML project  

-----------------------------------
## 📂 Project Structure

emotion-detection/  
│  
├── src/  
│   ├── train.py  
│   ├── realtime.py  
│  
├── dataset/  
├── models/  
├── app.py  
├── requirements.txt  
└── README.md  

-----------------------------------

## 🚀 How to Run

Clone repo:
          https://github.com/VedantVH/Real-Time-Facial-Emotion-Detection-System-Using-Deep-Learning.git
-----------------------------------

Setup: 
      python -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
-----------------------------------
Train:
      cd src
      python train.py
-----------------------------------
Run Real-Time:

python realtime.py
-----------------------------------

Launch Web App:

streamlit run app.py


-----------------------------------
## 🧪 Future Improvements

- Fine-tuning entire MobileNet layers  
- Model quantization for edge devices  
- Multi-face tracking  
- Cloud deployment (AWS / Streamlit Cloud)  
- Emotion analytics over time  

-----------------------------------

## 👨‍💻 Developer

Vedant VH  
AI & Deep Learning Enthusiast
