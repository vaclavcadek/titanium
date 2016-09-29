import numpy as np


class FullyConnectedNeuralNetworkEvaluator:
    def __init__(self, weights, biases, activations, norms=None):
        self._norms = norms
        self._weights = weights
        self._biases = biases
        self._activations = activations
        self._num_layers = len(self._weights)

    def predict_proba(self, X):
        a = X.copy()
        if self._norms:
            for i, fn in enumerate(self._norms):
                a[:, i] = fn(a[:, i])
        for i in range(self._num_layers):
            g = self._activations[i]
            W = self._weights[i]
            b = self._biases[i]
            a = g(np.dot(a, W.T) + b)
        return a

    def predict_classes(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, 1)
