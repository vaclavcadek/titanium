import numpy as np

logistic = lambda z: 1.0 / (1.0 + np.exp(-z))
tanh = lambda z: (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
rectifier = lambda z: np.maximum(0.0, z)