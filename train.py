import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, Input,
    BatchNormalization, Reshape,
    Bidirectional, LSTM
)

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

DATA_DIR = "dataset"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")

X = np.load(os.path.join(DATA_DIR, "X.npy"))
Y = np.load(os.path.join(DATA_DIR, "Y.npy"))
label_map = np.load(os.path.join(DATA_DIR, "labels.npy"),
                    allow_pickle=True).item()

print("X shape:", X.shape)   
print("Y shape:", Y.shape)

num_classes = len(label_map)

print("Total Classes:", num_classes)
X = (X - np.mean(X)) / np.std(X)
X = X[..., np.newaxis]

print("Reshaped X:", X.shape)
Y_cat = to_categorical(Y, num_classes)

with open(os.path.join(MODEL_DIR, "labels.pkl"), "wb") as f:
    pickle.dump(label_map, f)

print("Labels saved")

X_train, X_test, y_train, y_test = train_test_split(
    X, Y_cat,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Building CNN + BiLSTM Model...")

model = Sequential()

model.add(Input(shape=(100, 40, 1)))
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Reshape((-1, 128)))
model.add(Bidirectional(LSTM(128)))

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

lr_reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)
print("\nTraining Started...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[lr_reduce, early_stop]
)

model.save(os.path.join(MODEL_DIR, "cnn_bilstm_model.h5"))
print("\nModel Saved ")

loss, acc = model.evaluate(X_test, y_test)
print("\nFinal Accuracy:", acc)
print("Training Done ")