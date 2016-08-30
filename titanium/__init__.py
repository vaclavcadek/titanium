try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import numpy as np
from titanium import evaluation, activation

SUPPORTED_NS = {
    '4.2': 'http://www.dmg.org/PMML-4_2',
    '4.2.1': 'http://www.dmg.org/PMML-4_2',
    '4.3': 'http://www.dmg.org/PMML-4_3'
}


class PMMLVersionNotSupportedException(Exception):
    pass


def read_pmml(file):
    tree = ET.parse(file)
    pmml = tree.getroot()
    version = pmml.attrib.get('version', None)
    ns = SUPPORTED_NS.get(version, None)
    if None in [version, ns]:
        msg = 'Unsupported version of PMML.\nSupported versions are: {}'.format(SUPPORTED_NS.keys())
        raise PMMLVersionNotSupportedException(msg)
    nn = pmml.find('{}:NeuralNetwork'.format(version), SUPPORTED_NS)
    neural_inputs = nn.find('{}:NeuralInputs'.format(version), SUPPORTED_NS).findall('{}:NeuralInput'.format(version),
                                                                                     SUPPORTED_NS)
    norms = []
    for ni in neural_inputs:
        nc = ni.find('{}:DerivedField'.format(version), SUPPORTED_NS).find('{}:NormContinuous'.format(version), SUPPORTED_NS)
        ln = nc.findall('{}:LinearNorm'.format(version), SUPPORTED_NS)
        x0 = float(ln[0].attrib['orig'])
        y0 = float(ln[0].attrib['norm'])
        x1 = float(ln[1].attrib['orig'])
        y1 = float(ln[1].attrib['norm'])
        norms.append(lambda x: y0 + (x - x0) / (x1 - x0) * (y1 - y0))
    layers = nn.findall('{}:NeuralLayer'.format(version), SUPPORTED_NS)
    weights = []
    biases = []
    activations = [getattr(activation, l.attrib['activationFunction']) for l in layers]
    for l in layers:
        neurons = l.findall('{}:Neuron'.format(version), SUPPORTED_NS)
        biases.append(np.array([float(n.attrib['bias']) for n in neurons]))
        W = []
        for n in neurons:
            connections = n.findall('{}:Con'.format(version), SUPPORTED_NS)
            W.append([float(c.attrib['weight']) for c in connections])
        weights.append(np.array(W))
    return evaluation.FullyConnectedNeuralNetworkEvaluator(weights, biases, activations, norms)
