
import numpy as np
import cv2

def preprocess_image(img):
    """
    Preprocess image for emotion prediction
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = np.reshape(img, (1, 48, 48, 1))
    return img
