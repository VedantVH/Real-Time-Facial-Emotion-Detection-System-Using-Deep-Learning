import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import plotly.express as px
import pandas as pd
import os

# ------------------------------------
# PAGE CONFIG
# ------------------------------------
st.set_page_config(
    page_title="Emotion Intelligence Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Emotion Intelligence Dashboard")
st.markdown("Advanced Facial Emotion Recognition System")

# ------------------------------------
# LOAD MODEL
# ------------------------------------
@st.cache_resource
def load_emotion_model():
    return load_model("models/emotion_model.keras")

model = load_emotion_model()

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# ------------------------------------
# LOAD FACE CASCADE
# ------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)  # (1,48,48,3)
    return face_img

# ------------------------------------
# FILE UPLOAD
# ------------------------------------
uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", width=350)

    # Convert to grayscale ONLY for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        st.error("❌ No face detected. Please upload a clear frontal face image.")
    else:

        st.success(f"✅ Detected {len(faces)} face(s)")

        for (x, y, w, h) in faces:

            # Draw rectangle
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0,255,0), 2)

            face_roi = img_array[y:y+h, x:x+w]  # COLOR image

            processed_face = preprocess_face(face_roi)

            prediction = model.predict(processed_face, verbose=0)[0]

            max_index = np.argmax(prediction)
            emotion = emotion_labels[max_index]
            confidence = float(prediction[max_index])

            st.divider()

            # Always show result
            st.subheader("🎯 Prediction Result")
            st.success(f"Emotion: **{emotion}**")
            st.write(f"Confidence: **{confidence*100:.2f}%**")

            if confidence < 0.5:
                st.warning("⚠️ Low confidence prediction. Model may be uncertain.")

            # Probability chart
            df = pd.DataFrame({
                "Emotion": emotion_labels,
                "Probability": prediction
            })

            fig = px.bar(
                df,
                x="Emotion",
                y="Probability",
                title="Emotion Probability Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Show detected faces image
        st.image(img_array, caption="Detected Face(s)", width=350)

else:
    st.info("Upload an image to start emotion analysis.")