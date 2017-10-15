import unittest
import numpy as np
from titanium import read_pmml


class TreeIrisTest(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [4.8, 3.4, 1.6, 0.2],
            [4.9, 2.4, 3.3, 1.0],
            [6.5, 3.2, 5.1, 2.0]
        ])
        self.model = read_pmml('./resources/iris_tree.pmml', model='TreeModel')

    def test_predict_proba(self):
        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.9074074074074074, 0.09259259259259259],
            [0.0, 0.021739130434782608, 0.9782608695652174]
        ])
        probs = self.model.predict_proba(self.X)
        self.assertTrue(np.allclose(probs, expected), 'Output probabilities should be as expected.')

    def test_predict_classes(self):
        expected = np.array([0, 1, 2])
        preds = self.model.predict_classes(self.X)
        self.assertTrue(np.allclose(preds, expected), 'Output classes should be as expected.')
