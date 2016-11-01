import numpy as np
import titanium as ti

pmml = ti.read_pmml('iris_tree.pmml', model='TreeModel')
probs = pmml.predict_proba(np.array([3.0, 2.0, 4.0, 4.0]))
print(probs)
