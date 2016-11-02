import numpy as np
import titanium as ti
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data

pmml = ti.read_pmml('iris_tree.pmml', model='TreeModel')
probs = pmml.predict_proba(X)
print(probs)
