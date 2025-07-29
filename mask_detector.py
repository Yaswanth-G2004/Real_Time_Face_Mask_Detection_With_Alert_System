import cv2
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound
import threading

# Load your trained model
model = load_model("mask_detector_model.h5")

# Labels and colors
labels_dict = {0: 'Mask', 1: 'No Mask'}
colors_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to play sound without freezing webcam
def play_alert():
    playsound('mixkit-appliance-ready-beep-1076.wav')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (128, 128))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 128, 128, 3)

        result = model.predict(reshaped)[0]
        label = np.argmax(result)
        color = colors_dict[label]
        label_text = labels_dict[label]

        # Draw rectangle and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # If "No Mask", play alert sound
        if label == 1:
            threading.Thread(target=play_alert, daemon=True).start()

    # Show the frame
    cv2.imshow('Face Mask Detection - Live Alert', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
