titanium
========

Installation
------------

To install titanium, simply:

.. code-block:: bash

    $ pip install titanium

Example
-------

Example on Iris data - for more examples see the examples folder.

.. code-block:: python

    from keras2pmml import keras2pmml
    from sklearn.datasets import load_iris
    import numpy as np
    import theano
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import accuracy_score
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers.core import Dense
    from sklearn.preprocessing import StandardScaler

    import titanium as ti
    import os

    iris = load_iris()
    X = iris.data
    y = iris.target

    theano.config.floatX = 'float32'
    X = X.astype(theano.config.floatX)
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

    y_train_ohe = np_utils.to_categorical(y_train)
    y_test_ohe = np_utils.to_categorical(y_test)

    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train)
    X_test_scaled = std.transform(X_test)
    model = Sequential()
    model.add(Dense(input_dim=X_train_scaled.shape[1], output_dim=100, activation='tanh'))
    model.add(Dense(input_dim=20, output_dim=20, activation='tanh'))
    model.add(Dense(input_dim=5, output_dim=y_test_ohe.shape[1], activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    model.fit(X_train_scaled, y_train_ohe, nb_epoch=100, batch_size=1, verbose=3, validation_data=None)

    params = {
        'copyright': 'Václav Čadek',
        'description': 'Simple Keras model for Iris dataset.',
        'model_name': 'Iris Model'
    }

    keras2pmml(model, file='iris.pmml', **params)
    pmml = ti.read_pmml('iris.pmml')
    os.unlink('iris.pmml')

    keras_preds = model.predict_classes(X_test_scaled)
    titanium_preds = pmml.predict_classes(X_test_scaled)

    print('Accuracy (Keras): {accuracy}'.format(accuracy=accuracy_score(y_test, keras_preds)))
    print('Accuracy (Titanium): {accuracy}'.format(accuracy=accuracy_score(y_test, titanium_preds)))



What is supported?
------------------
- Models
    * keras.models.Sequential
- Activation functions
    * tanh
    * sigmoid/logistic

License
-------

This software is licensed under MIT licence.

- https://opensource.org/licenses/MIT