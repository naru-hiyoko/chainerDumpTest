import chainer
from chainer import functions as F
from chainer import links as L

import numpy as np

class TinyNet(chainer.Chain):
    def __init__(self):
        super(TinyNet, self).__init__(
            conv1_1=L.Convolution2D(1, 16, 3, stride=1, pad=0),
            conv1_2=L.Convolution2D(16, 32, 3, stride=1, pad=0),

            deconv2_1=L.Deconvolution2D(32, 16, 5, stride=1, pad=1),
            deconv2_2=L.Deconvolution2D(16, 1, 5, stride=1, pad=1),

            conv3_1 = L.Convolution2D(32, 16, 3, stride=2),
            fc3_1 = L.Linear(400, 10),
        )

        self.mean = np.asarray([104, 117, 124], dtype=np.float32)


    def __call__(self, x):
        layers = {}

        layer_namesA = ['conv1_1', 'conv1_2', 'deconv2_1', 'deconv2_2']

        #pathA
        h = x
        for name in layer_namesA:
            if "conv" in name:
                h = self[name](h)
                layers[name] = h

        #pathB
        layer_namesB = ['pool', 'conv3_1', 'fc3_1']
        h = layers["conv1_2"]
        for name in layer_namesB:
            if 'pool' in name:
                h = F.max_pooling_2d(h, 2)
            else:
                h = self[name](h)
            layers[name] = h


        return layers


class TinyNetFC(chainer.Chain):
    def __init__(self):
        super(TinyNetFC, self).__init__(
            fc1=L.Linear(4, 3),
        )

    def __call__(self, x):
        h = x
        layers = dict()

        for name in ['fc1']:
            h = F.relu(self[name](h))
            layers[name] = h

        return layers
