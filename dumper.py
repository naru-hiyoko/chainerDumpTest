import numpy as np
import chainer.serializers
from chainer import Variable
import chainer.functions as F
import chainer.links as L

import json

"""
;;; in Chain Object...

    def __call__(self, x):
        layer_names = ['conv1_1', 'conv1_2', 'deconv2_1', 'deconv2_2', 'conv3_1', 'fc3_1']
        layers = {}
        h = x
        for layer_name in layer_names:
            #print(layer_name)

            if layer_name == 'pool':
                h = F.max_pooling_2d(h, 2, stride=2)
            elif 'conv' in layer_name:
                if layer_name == 'conv3_1':
                    h = F.relu(layers['conv1_2'])
                    layers[layer_name] = h
                else:
                    h = F.relu(self[layer_name](h))
                    layers[layer_name] = h
            elif 'deconv' in layer_name:
                h = F.relu(self[layer_name](h))
                layers[layer_name] = h
            elif 'fc' in layer_name:
                h = F.relu(self[layer_name](h))
                layers[layer_name] = h

        return layers

;;; represents graph by Chain Object

x = Variable(test[0].reshape(1, 1, 28, 28))

layers = net(x)
loss = Variable(np.zeros([]).astype(np.float32))

inputs = [x]

;;; outputs you want.
outputs = [layers['deconv2_2'], layers['fc3_1']]
json_content = dumper.graphRepresentation(loss, inputs, outputs)
#print(json_content)

"""

def setValue(jsonContent, variable, forKey=''):
    content = dict()
    content['data'] = [str(x) for x in variable.data.reshape(-1)]
    content['shape'] = variable.data.shape
    jsonContent[forKey] = content
    return jsonContent

def setInOut(content, creator):
    for val in creator.inputs:
        a = [str(id(x)) if x.name is None else x.name for x in creator.inputs]
        content = setVariable(content, val)
        content['inputs'] = a
    for val in creator.outputs:
        a = [str(id(x())) if x().name is None else x().name for x in creator.outputs]
        content = setVariable(content, val())
        content['outputs'] = a
    return content

def setVariable(content, variable):
    if variable.name is None:
        variable.name = str(id(variable))

    tag1 = '{}_data'.format(variable.name)
    tag2 = '{}_shape'.format(variable.name)
    tag3 = '{}_grad'.format(variable.name)
    if not variable.data is None:
        content[tag1] = [str(x) for x in variable.data.reshape(-1)]
        content[tag2] = variable.data.shape
    if not variable.grad is None:
        content[tag3] = [str(x) for x in variable.grad.reshape(-1)]


    return content

def setConv(content, creator):
    c = creator
    content['params'] = [c.sx, c.sy, c.pw, c.ph]
    content = setInOut(content, creator)
    return content

def setLinear(content, creator):
    return setInOut(content, creator)

def setPool(content, creator):
    c = creator
    content['params'] = [c.kw, c.kh, c.sx, c.sy, c.pw, c.ph]
    return setInOut(content, creator)

def setReLU(content, creator):
    return setInOut(content, creator)


def dump(loss):
    layer_contents = dict()

    def recursion(loss):
        creator = loss.creator
        className = creator.__class__.__name__

        if creator == None:
            return

        inputs = creator.inputs
        outputs = creator.outputs

        content = dict()
        content['type'] = className
        content['id'] = str(id(creator))
        content['inputs'] = [str(id(x)) for x in inputs]
        content['outputs'] = [str(id(x())) for x in outputs]

        print(content)

        if className == "Convolution2DFunction":
            content = setConv(content, creator)
        elif className == "Deconvolution2DFunction":
            content = setConv(content, creator)
        elif className == 'MaxPooling2D' or className == 'AvgPooling2D':
            content = setPool(content, creator)
        elif className == "LinearFunction":
            content = setLinear(content, creator)
        elif className == "ReLU":
            content = setReLU(content, creator)
            pass
        elif className == "AddConstant":
            content = setInOut(content, creator)
        elif className == "Add":
            content = setInOut(content, creator)
        
        elif className == "MulConstant":
            content = setInOut(content, creator)
        elif className == "Mul":
            content = setInOut(content, creator)
        elif className == "PowVarConst":
            content = setInOut(content, creator)
        elif className == "Sub":
            content = setInOut(content, creator)
            
        else:
            print('warning: ignore {}'.format(className))
            pass

        layer_contents[str(id(creator))] = content

        for input in inputs:
            recursion(input)

    recursion(loss)

    return layer_contents


def graphRepresentation(loss, input_variables, end_layer_links):
    json_content = dump(loss)

    inputs = [str(id(x)) for x in input_variables]
    outputs = [str(id(x)) for x in end_layer_links]
    print('inputs : {}'.format(inputs))
    print('outputs : {}'.format(outputs))

    json_content['inputs'] = inputs
    json_content['outputs'] = outputs

    return json_content


