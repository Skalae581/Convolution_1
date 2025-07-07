# -*- coding: utf-8 -*-
"""
Optimierter Grid Search: LFW Faces CNN
@author: TAKO
"""

from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, callbacks, optimizers
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

# ==================== Daten laden (kleiner + effizienter)
lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4, color=True)
X = lfw_people.images.astype(np.float16)  # Speicher sparen
y = lfw_people.target
class_names = lfw_people.target_names
num_classes = len(class_names)

print("Klassen:", class_names)
print("Verteilung:\n", pd.Series(y).value_counts())

y_encoded = to_categorical(y, num_classes)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y, random_state=42
)

# ==================== TensorBoard
log_dir = os.path.join("logs", "gridsearch", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# ==================== Grid Search Parameter
learning_rates = [1e-4, 1e-3, 1e-2]
dense_units_list = [64, 128, 256]

results = []
early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# ==================== Modell-Funktion
def build_model(lr, dense_units):
    model = models.Sequential([
        layers.Input(shape=X_train.shape[1:]),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==================== Grid Search Schleife
for lr in learning_rates:
    for dense_units in dense_units_list:
        print(f"\n--- Training mit learning_rate={lr}, dense_units={dense_units} ---")

        model = build_model(lr, dense_units)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=8,   # Kleiner für weniger RAM
            callbacks=[early_stopping, tensorboard_cb],
            verbose=0
        )

        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Accuracy: {val_acc:.4f}")

        results.append({
            "learning_rate": lr,
            "dense_units": dense_units,
            "val_accuracy": val_acc
        })

# ==================== Beste Kombination
results = sorted(results, key=lambda x: -x["val_accuracy"])
print("\nBeste Parameterkombination:", results[0])

# ==================== Bestes Modell nochmal trainieren
best_lr = results[0]['learning_rate']
best_units = results[0]['dense_units']
print(f"\nTraining Bestes Modell: learning_rate={best_lr}, dense_units={best_units}")

best_model = build_model(best_lr, best_units)

history = best_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=8,
    callbacks=[early_stopping, tensorboard_cb],
    verbose=0
)

# ==================== Lernkurve
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Lernkurve (Bestes Modell)")
plt.show()

# ==================== Confusion Matrix
y_pred = best_model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)
y_val_labels = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_val_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# ==================== Ergebnis
val_loss, val_acc = best_model.evaluate(X_val, y_val)
print(f"\nFinale Val Accuracy (Bestes Modell): {val_acc:.4f}")
