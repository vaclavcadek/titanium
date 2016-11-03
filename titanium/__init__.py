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

SUPPORTED_MODELS = [
    'NeuralNetwork',
    'TreeModel'
]


class PMMLVersionNotSupportedException(Exception):
    pass


def read_pmml(file, model='NeuralNetwork'):
    tree = ET.parse(file)
    pmml = tree.getroot()
    version = pmml.attrib.get('version', None)
    ns = SUPPORTED_NS.get(version, None)
    if None in [version, ns]:
        msg = 'Unsupported version of PMML.\nSupported versions are: {}'.format(SUPPORTED_NS.keys())
        raise PMMLVersionNotSupportedException(msg)

    if model == 'NeuralNetwork':
        nn = pmml.find('{}:NeuralNetwork'.format(version), SUPPORTED_NS)
        neural_inputs = nn.find('{}:NeuralInputs'.format(version), SUPPORTED_NS).findall(
            '{}:NeuralInput'.format(version),
            SUPPORTED_NS)
        norms = []
        for ni in neural_inputs:
            nc = ni.find('{}:DerivedField'.format(version), SUPPORTED_NS).find('{}:NormContinuous'.format(version),
                                                                               SUPPORTED_NS)
            ln = nc.findall('{}:LinearNorm'.format(version), SUPPORTED_NS)
            x0 = float(ln[0].attrib['orig'])
            y0 = float(ln[0].attrib['norm'])
            x1 = float(ln[1].attrib['orig'])
            y1 = float(ln[1].attrib['norm'])
            norms.append(lambda x, x0=x0, y0=y0, x1=x1, y1=y1: y0 + (x - x0) / (x1 - x0) * (y1 - y0))
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
    elif model == 'TreeModel':
        decision_tree = pmml.find('{}:TreeModel'.format(version), SUPPORTED_NS)
        mining_schema = decision_tree.find('{}:MiningSchema'.format(version), SUPPORTED_NS)
        mining_fields = mining_schema.findall('{}:MiningField'.format(version), SUPPORTED_NS)
        root = decision_tree.find('{}:Node'.format(version), SUPPORTED_NS)
        fields = [f.attrib['name'] for f in mining_fields if f.attrib.get('usageType', None) != 'predicted']

        def get_children(node):
            return node.findall('{}:Node'.format(version), SUPPORTED_NS)

        def get_predicate(node):
            return node.find('{}:SimplePredicate'.format(version), SUPPORTED_NS)

        def get_score_distributions(node):
            return node.findall('{}:ScoreDistribution'.format(version), SUPPORTED_NS)

        return evaluation.TreeModelEvaluator(fields, root, get_children, get_predicate, get_score_distributions)

    elif model == 'Segmentation':
        mining_model = pmml.find('{}:MiningModel'.format(version), SUPPORTED_NS)
        mining_schema = mining_model.find('{}:MiningSchema'.format(version), SUPPORTED_NS)
        segmentation = mining_model.find('{}:Segmentation'.format(version), SUPPORTED_NS)
        mining_fields = mining_schema.findall('{}:MiningField'.format(version), SUPPORTED_NS)
        segments = []
        for segment in segmentation.findall('{}:Segment'.format(version), SUPPORTED_NS):
            decision_tree = segment.find('{}:TreeModel'.format(version), SUPPORTED_NS)
            root = decision_tree.find('{}:Node'.format(version), SUPPORTED_NS)
            fields = ['x{}'.format(i) for i in range(1, 13)]

            def get_children(node):
                return node.findall('{}:Node'.format(version), SUPPORTED_NS)

            def get_predicate(node):
                return node.find('{}:SimplePredicate'.format(version), SUPPORTED_NS)

            def get_score_distributions(node):
                return node.findall('{}:ScoreDistribution'.format(version), SUPPORTED_NS)

            evaluator = evaluation.TreeModelEvaluator(fields, root, get_children, get_predicate, get_score_distributions)
            segments.append(evaluator)

        return evaluation.SegmentationEvaluator(segments)
