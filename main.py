import numpy as np
import net

from chainer.datasets import mnist
from chainer.optimizers import adam
import chainer.serializers
from chainer import Variable
import chainer.functions as F
import chainer.links as L

import json

import skimage
from skimage.io import imsave

from chainer.datasets import get_mnist
(train, test) = get_mnist(ndim=3)

model = net.TinyNet()

#optim = adam.Adam()
#optim.setup(model)

_in = Variable(train[0][0].reshape([1, 1, 28, 28]))
layers = model(_in)
_lossA = F.mean_squared_error(layers['deconv2_2'], _in)
_lossB = F.softmax_cross_entropy(layers['fc3_1'], Variable(np.asarray([1]).astype(np.int32)))
print(_lossA.data)
print(_lossB.data)

_loss = _lossA + _lossB

#print(_loss.creator)

import dumper

content = dumper.graphRepresentation(_loss, [_in], [])

#with open('/Volumes/ramdisk/model.json', 'w') as fp:
#    json.dump(content, fp)


