import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

data_dir = "Sign_languages"
img_size = 64

valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
X, y = [], []

labels = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
print(f"Classes: {labels}")

for label_id, label in enumerate(labels):
    folder_path = os.path.join(data_dir, label)
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(valid_exts):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label_id)
                count += 1
    print(f"  {label}: {count} images")

X = np.array(X, dtype="float32") / 255.0
y = to_categorical(np.array(y))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                              height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    BatchNormalization(), MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),
    Flatten(), Dense(256, activation='relu'), Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val), epochs=25)

model.save("sign_model.h5")
joblib.dump(labels, "labels.pkl")
print("Model and labels saved!")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.plot(history.history['accuracy'], label='train_acc')
ax1.plot(history.history['val_accuracy'], label='val_acc')
ax1.set_title('Accuracy'); ax1.legend()
ax2.plot(history.history['loss'], label='train_loss')
ax2.plot(history.history['val_loss'], label='val_loss')
ax2.set_title('Loss'); ax2.legend()
plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
