#Chainer Dump Test

## Chainer の学習パラメータを書き出します


### ネットワークの定義
レイヤーの dict を用意する
 
```|swfit| 
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

```

### 入力
```
x = Variable(test[0].reshape(1, 1, 28, 28))
inputs = [x]

```

### 出力
```
layers = net(x)
loss = Variable(np.zeros([]).astype(np.float32))
outputs = [layers['deconv2_2'], layers['fc3_1']]
```

### 書き出し
```
content = dumper.graphRepresentation(loss, _in, _out)
with open('/Volumes/ramdisk/model.json', 'w') as fp:
    json.dump(content, fp)

```

### 最後に
[Chainer のチュートリアルを読んで](http://docs.chainer.org/en/latest/tutorial/basic.html)
