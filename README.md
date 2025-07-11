﻿# Convolution_1
 # Projekt: CNN Klassifikation von LFW-Personenbildern

Dieses Projekt trainiert ein **Convolutional Neural Network (CNN)** zur Klassifikation von Bildern prominenter Personen aus dem **Labeled Faces in the Wild (LFW)**-Datensatz.

---

## Inhalt

* Laden des **LFW-People Datensatzes** mit mindestens 50 Bildern pro Person
* Visualisierung einiger Beispielbilder
* Vorverarbeitung der Bilder:

  * Zuschneiden (slice)
  * Normalisierung
  * One-Hot-Encoding der Labels
* Aufteilung in Trainings- und Validierungsdaten
* Aufbau eines CNN mit mehreren Conv2D- und MaxPooling-Schichten
* Training mit EarlyStopping
* Anzeige der Lernkurven
* Evaluation der Genauigkeit

---

## Verwendete Bibliotheken

* TensorFlow / Keras
* scikit-learn
* NumPy
* Matplotlib
* Pandas

---

## Code-Ausschnitte

### Datensatz laden:

```python
lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=1, color=True,
                              slice_=(slice(50, 200, None), slice(50, 200, None)))
```

### CNN-Architektur:

```python
model = models.Sequential()
model.add(layers.Input(shape=X_train.shape[1:]))
model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
# Weitere Conv2D und MaxPooling-Schichten folgen
```

### Kompilierung und Training:

```python
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=300, batch_size=32,
                    validation_data=(X_val, y_val), callbacks=[early_stopping])
```

---

## Ergebnis

Die finale **Validierungsgenauigkeit (val\_accuracy)** wird am Ende ausgegeben.
Zusätzlich werden die Lernkurven für **Accuracy** im Verlauf des Trainings angezeigt.

---

## Start des Trainings

1. Python-Datei im gewünschten Ordner ablegen
2. Ausführen: z.B. in Spyder, PyCharm oder Jupyter Notebook
3. TensorFlow/Scikit-Learn müssen installiert sein:

```bash
pip install tensorflow scikit-learn matplotlib pandas
```

---

## Autor

TAKO, 07.07.2025

