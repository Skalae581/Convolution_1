# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 12:33:27 2025

@author: TAKO
"""

from sklearn.datasets import fetch_lfw_people
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
# Lade nur Bilder von Personen mit mindestens 50 Bildern
lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=1, color=True,slice_=(slice(50, 200, None), slice(50, 200, None)))

fig, axes = plt.subplots(2, 5, figsize=(250, 250))
for i, ax in enumerate(axes.flat):
    ax.imshow(lfw_people.images[i])
    ax.set_title(lfw_people.target_names[lfw_people.target[i]])
    ax.axis('off')
plt.show()
X = lfw_people.images # Werte zwischen 0 und 1
y = lfw_people.target
class_names = lfw_people.target_names

print("Klassen:", lfw_people.target_names)
print("Klassenanzahl:", len(lfw_people.target_names))
print("Verteilung:\n", pd.Series(y).value_counts())


# ============================
# 3. Labels OneHot encoden
# ============================
y_encoded = to_categorical(y, num_classes=len(class_names))

# ============================
# 4. Trainings- und Validierungsdaten aufteilen
# ============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y, random_state=42
)

# ============================
# 5. CNN Modell erstellen
# ============================
model = models.Sequential()
model.add(layers.Input(shape=X_train.shape[1:]))

# Erste Convolution-Schicht mit 32 Filtern (3x3 Größe), ReLU-Aktivierung
# padding="same" bedeutet, dass die Ausgabe die gleiche Größe wie die Eingabe behält
model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
# MaxPooling reduziert die Bildgröße um die Hälfte (2x2 Bereich)
model.add(layers.MaxPooling2D((2, 2)))

# Zweite Convolution + Pooling
model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))

# Dritte Convolution + Pooling
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))

# =============================================================================
# # Dritte Convolution + Pooling
# model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
# model.add(layers.MaxPooling2D((2, 2)))
# 
# =============================================================================
# Übergang zur Dense-Schicht
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(len(class_names), activation="softmax"))

model.summary()

# ============================
# 6. Kompilieren
# ============================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================
# 7. Callbacks (optional)
# ============================
early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# ============================
# 8. Training
# ============================
history = model.fit(
    X_train, y_train,
    #epochs=1000,
    epochs=300,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# ============================
# 9. Lernkurven anzeigen
# ============================
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Lernkurve")
plt.show()

# ============================
# 10. Ergebnis: Genauigkeit
# ============================
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"\nVal Accuracy: {val_acc:.4f}")