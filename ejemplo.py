"""
example_ml_libraries.py

Este script muestra cómo importar y usar librerías populares:
- PyTorch (deep learning)
- TensorFlow (deep learning)
- Plotly (visualización)
- NumPy (cálculo numérico)

Autor: Ejemplo educativo
"""

# =========================
# IMPORTS
# =========================

import numpy as np

# PyTorch
import torch
import torch.nn as nn

# TensorFlow
import tensorflow as tf

# Plotly
import plotly.graph_objects as go


# =========================
# EJEMPLO 1: NUMPY
# =========================

print("\n=== NUMPY ===")
array = np.array([1, 2, 3, 4])
print("Array:", array)
print("Media:", np.mean(array))


# =========================
# EJEMPLO 2: PYTORCH
# =========================

print("\n=== PYTORCH ===")

# Crear tensor
tensor = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", tensor)

# Operación simple
print("Tensor * 2:", tensor * 2)


# Modelo simple en PyTorch
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


model = SimpleModel()
input_tensor = torch.randn(1, 3)
output = model(input_tensor)

print("Salida del modelo PyTorch:", output)


# =========================
# EJEMPLO 3: TENSORFLOW
# =========================

print("\n=== TENSORFLOW ===")

# Crear tensor
tf_tensor = tf.constant([1.0, 2.0, 3.0])
print("TensorFlow Tensor:", tf_tensor)

# Operación
print("TensorFlow * 2:", tf_tensor * 2)


# Modelo simple en TensorFlow
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(3,))
])

tf_output = tf_model(tf.random.normal((1, 3)))
print("Salida del modelo TensorFlow:", tf_output)


# =========================
# EJEMPLO 4: PLOTLY
# =========================

print("\n=== PLOTLY ===")

x = [1, 2, 3, 4]
y = [10, 15, 13, 17]

fig = go.Figure(
    data=go.Scatter(x=x, y=y, mode='lines+markers')
)

fig.update_layout(
    title="Ejemplo de gráfica con Plotly",
    xaxis_title="X",
    yaxis_title="Y"
)

# Mostrar gráfica (abre en navegador)
fig.show()


# =========================
# FIN
# =========================
print("\nScript ejecutado correctamente 🚀")