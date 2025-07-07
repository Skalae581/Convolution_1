# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:31:41 2022

@author: Bernd Ebenhoch
"""

# Wir wollen farbige Bilder von Prominenten aus dem lFW-People-Datensatz von scikit-learn klassifizieren
# Bilder herunterladen
# Einige Bilder darstellen
# Bilder normalisieren
# Prüfen wie viele Labels es gibt und wie oft diese vertreten sind
# Labels onehotencoden
# Daten in Trainings- und Validierungsdaten aufteilen

# Ein Convolution-Neuronales-Netz designen
# Ein Sequential-Model erzeugen
# Die erste Schicht kann eine Input-Schicht sein, um dem Modell die Größe der Merkmale mitzuteilen und model.summary() anzeigen zu können
# Zum Modell eine Conv2D-Schicht hinzufügen mit einer bestimmten Anzahl von Filtern
# Anschließend folgt eine MaxPooling2D-Schicht
# Conv2D und MaxPooling wiederholen, dabei jeweils Conv2D
# und MaxPooling2D mehrmals in den Conv2D-Schichten die Anzahl
# der Filter vergrößern bis die Anzahl der Informationen genügend
# reduziert wurde z. B. auf ca. 1500 (Höhe x Breite x Kanäle)

# Anschließend eine Flatten-Schicht hinzufügen (alternativ GlobalAveragePooling)
# Anschließend eine Dense-Schicht als Zwischenschicht hinzufügen
# Anschließend eine Dense-Schicht für die Multiklassenklassifikation als Ausgabeschicht hinzufügen

# Das Modell mit passender Loss-Funktion und Metrik compilieren
# Das Modell fitten
# Die Ergebnisse beurteilen und Hyperparameter optimieren


from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Die Bilder herunterladen
data = fetch_lfw_people(data_home=None,
                        funneled=True,
                        resize=1,  # Im Falle von Arbeitsspeicherproblemen: 0.5
                        min_faces_per_person=50,
                        color=True,
                        slice_=(slice(50, 200, None), slice(50, 200, None)),
                        #slice_=(slice(0, 250, None), slice(0, 250, None)),

                        download_if_missing=True,
                        return_X_y=False)

# In x, y aufteilen
x = data.images
y = data.target


# Bilder normalisieren
# Je nach scikit-learn-Version ist x.max() entweder 1 oder 255
# Wir geben uns erst mal die Max- und Min-Werte der Pixelintensitäten aus
print('Max and Min pixel values:', x.max(), x.min())

# Die Pixel sollten am besten alle mit dem gleichen Faktor skaliert werden
# Mit dem MinMaxscaler würden wir pro Pixel einen unterschielichen Skalierungsfaktor
# bekommen und dadurch werden die Bilder sehr scheckig
# manche Ecken oder Kanten würden in andere Bilder einfließen
x = x/x.max()

# Einige Bilder darstellen
for i in range(12):
    plt.subplot(3, 4, i+1)  # Subplot mit 3 Zeilen und 4 Spalten
    plt.imshow((x[y == i][0]))
    plt.title(data.target_names[i], fontsize=8)
    plt.axis('off')  # Keine Achsenbeschriftung
plt.show()
