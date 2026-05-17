import cv2
import numpy as np
import joblib
import pyttsx3
from tensorflow.keras.models import load_model

model = load_model("sign_model.h5")
labels = joblib.load("labels.pkl")

engine = pyttsx3.init()
engine.setProperty('rate', 145)

cap = cv2.VideoCapture(0)
img_size = 64
last_label = None
cooldown = 25
counter = 0

print("Press 'q' to quit.")
print("Place your hand inside the green box.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    x1, y1, x2, y2 = w - 320, 100, w - 50, 370
    roi = frame[y1:y2, x1:x2]
    hand = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    hand = cv2.resize(hand, (img_size, img_size))
    hand = hand.astype("float32") / 255.0
    hand = np.expand_dims(hand, axis=0)
    preds = model.predict(hand, verbose=0)
    label_index = np.argmax(preds)
    confidence = np.max(preds)
    label = labels[label_index]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({confidence:.2f})", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if confidence > 0.8:
        if label != last_label or counter >= cooldown:
            engine.say(label)
            engine.runAndWait()
            last_label = label
            counter = 0
    counter += 1
    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
