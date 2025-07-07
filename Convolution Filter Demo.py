# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 08:47:35 2022

@author: Bernd Ebenhoch
"""


# In diesem Beispiel wollen wir beobachten wie zwei verschiedene
# Convolution-Filter auf ein Bild wirken und
# den Einfluss der Hyperparameter der Conv2D-Schicht erkunden
# In diesem Beipiel führen wir kein Training der Convolution-Schichten
# durch.

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

size = 20


# Numpy mit Nullen der Größe (size, size) erzeugen
# Und eine H-form mit Einsen zeichnen
x = np.zeros((size, size))
x[5:16, 6] = 1
x[5:16, 13] = 1
x[10, 6:13] = 1


print(x.shape)


# Die Matrix mit dem imshow-Befehl in Matplotlib als Graustufenbild darstellen
plt.imshow(x, cmap='gray', interpolation=None)
plt.title('Original')
plt.show()


# Anzahl der Datenpunkte, Höhe, Breite, Anzahl der Farbkanäle
x = x.reshape(1, size, size, 1)


print(x.shape)

# Wr können eine Initialisierungsroutine definieren


def custom_weights(shape, dtype=None):
    print(shape)
    # vertikaler Filter und horizontaler Filter
    kernel = np.array([[[[0., 0.]], [[0., 1.]], [[0., 0.]]],
                       [[[1., 0.]], [[1., 1.]], [[1., 0.]]],
                       [[[0., 0.]], [[0., 1.]], [[0., 0.]]]])

    # kernel = tf.Variable(kernel, dtype=tf.float32)
    print(kernel.shape)
    print(kernel[:, :, 0, 0])
    print(kernel[:, :, 0, 1])

    return kernel


# Eine Conv2D-Schicht definieren
layer = keras.layers.Conv2D(2, (3, 3), padding='valid',
                            strides=(1, 1),
                            dilation_rate=(1, 1),
                            kernel_initializer=custom_weights,
                            # activation='relu'
                            )


# die Schicht auf x anwenden und das Ergebnis als NumPy-Array zurückgeben
y = layer(x).numpy()

# y = keras.layers.MaxPool2D()(y)
# y = y.numpy()

print(layer.weights[0].numpy())

print(x.shape)
print(y.shape)

plt.imshow(y[0, :, :, 0], cmap='gray', interpolation='none')
plt.title('Horizontaler Filter')
plt.show()
plt.imshow(y[0, :, :, 1], cmap='gray', interpolation='none')
plt.title('Vertikaler Filter')
plt.show()
