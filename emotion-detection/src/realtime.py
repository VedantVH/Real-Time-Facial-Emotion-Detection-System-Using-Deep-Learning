import cv2
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from tensorflow.keras.models import load_model

model = load_model("../models/emotion_model.keras")

emotion_labels = [
    "Angry","Disgust","Fear",
    "Happy","Neutral","Sad","Surprise"
]

emotion_history = deque(maxlen=10)
emotion_count = defaultdict(int)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

start_time = time.time()
frame_count = 0

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for i,(x,y,w,h) in enumerate(faces):

        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi,(48,48))
        roi = roi.astype("float")/255.0
        roi = np.reshape(roi,(1,48,48,3))

        prediction = model.predict(roi,verbose=0)[0]
        max_index = np.argmax(prediction)

        raw_emotion = emotion_labels[max_index]
        confidence = round(prediction[max_index]*100,2)

        emotion_history.append(raw_emotion)
        emotion = max(set(emotion_history), key=emotion_history.count)

        emotion_count[emotion]+=1

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,f"ID {i+1}",(x,y-30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        cv2.putText(frame,f"{emotion} ({confidence}%)",
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)

    fps = frame_count/(time.time()-start_time)
    cv2.putText(frame,f"FPS: {round(fps,2)}",
                (10,frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,(0,255,255),2)

    cv2.imshow("Advanced Emotion AI",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save Session JSON
session_data={
    "duration":round(time.time()-start_time,2),
    "emotion_summary":dict(emotion_count)
}

with open("../emotion_session_log.json","w") as f:
    json.dump(session_data,f,indent=4)

# Emotion Graph
if emotion_count:
    plt.bar(emotion_count.keys(),emotion_count.values())
    plt.title("Emotion Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../emotion_summary.png")
    plt.show()

print("Session Saved.")
