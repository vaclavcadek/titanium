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


class TreeModelEvaluator:
    def __init__(self, fields, root, get_children, get_predicate, get_score_distributions):
        self._fields = fields
        self._root = root
        self._get_children = get_children
        self._get_predicate = get_predicate
        self._get_score_distributions = get_score_distributions

    def predict_proba(self, X):
        probs = []
        for i in range(X.shape[0]):
            node = self._root
            while True:
                children = self._get_children(node)
                # is leaf
                if len(children) == 0:
                    break
                else:
                    p = self._get_predicate(children[0])
                    field = p.attrib['field']
                    operator = p.attrib['operator']
                    split_point = float(p.attrib['value'])
                    val = X[i, self._fields.index(field)]
                    condition = {
                        'greaterThan': val > split_point,
                        'greaterOrEqual': val >= split_point,
                        'lessThan': val < split_point,
                        'lessOrEqual': val <= split_point,
                    }[operator]
                    if condition:
                        node = children[0]
                    else:
                        node = children[1]
            record_counts = [float(c.attrib['recordCount']) for c in self._get_score_distributions(node)]
            probs.append(np.array(record_counts) / np.sum(record_counts))
        return np.array(probs)

    def predict_classes(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, 1)


class SegmentationEvaluator:
    def __init__(self, segment_evaluators):
        self._segment_evaluators = segment_evaluators

    def predict_proba(self, X):
        probs = np.zeros(self._segment_evaluators[0].predict_proba(X).shape)
        individual_predictions = [t.predict_proba(X) for t in self._segment_evaluators]
        for predictions in individual_predictions:
            probs += predictions
        return np.array(probs) / len(individual_predictions)

    def predict_classes(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, 1)
