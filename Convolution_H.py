# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 11:22:54 2025

@author: TAKO
"""
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# Bild erzeugen (eine H-Form in 20x20)
# ============================================
size = 20
x = np.zeros((size, size))
x[5:16, 6] = 1
x[5:16, 13] = 1
x[10, 6:13] = 1

# Plot des Originalbilds
plt.imshow(x, cmap='gray')
plt.title('Original H')
plt.show()

# Tensor-Format für Conv2D
x = x.reshape(1, size, size, 1)

# ============================================
# Eigener Kernel (Laplace-ähnlich)
# ============================================
def create_kernel():
    kernel = np.array([
        [[[0.]], [[1.]], [[0.]]],
        [[[1.]], [[-4.]], [[1.]]],
        [[[0.]], [[1.]], [[0.]]]
    ], dtype=np.float32)  # Form: (3, 3, 1, 1)
    return kernel

# Kernel in korrekter Form für TensorFlow
kernel_weights = create_kernel().reshape(3, 3, 1, 1)

# ============================================
# Conv2D-Layer mit Custom-Kernel erstellen
# ============================================
layer = keras.layers.Conv2D(
    filters=1,
    kernel_size=(3, 3),
    padding='valid',
    use_bias=False  # Kein Bias, damit nur der Filter wirkt
)

# Layer aufbauen und eigene Gewichte setzen
layer.build(input_shape=(None, size, size, 1))
layer.set_weights([kernel_weights])

# ============================================
# Faltung anwenden
# ============================================
y = layer(x).numpy()

# ============================================
# Ergebnis anzeigen
# ============================================
plt.imshow(y[0, :, :, 0], cmap='gray', interpolation='none')
plt.title('Gefiltertes Bild (Laplace)')
plt.show()
