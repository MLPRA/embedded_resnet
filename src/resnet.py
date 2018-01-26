import chainer
import collections
import numpy as np
from chainer import Variable
from chainer.dataset import concat_examples
from chainer.links.model.vision.resnet import prepare
from chainer.functions.activation.relu import relu
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d


class ResNet49Layers(chainer.links.ResNet50Layers):
    def __call__(self, x, layers=['res5'], kwargs={}):
        return super().__call__(x, layers=layers, **kwargs)

    @property
    def functions(self):
        return collections.OrderedDict([
            ('conv1', [self.conv1, self.bn1, relu]),
            ('pool1', [lambda x: max_pooling_2d(x, ksize=3, stride=2)]),
            ('res2', [self.res2]),
            ('res3', [self.res3]),
            ('res4', [self.res4]),
            ('res5', [self.res5])
        ])

    def _layer_out(self, images, layer):
        layers = ['conv1', 'bn1', 'res2', 'res3', 'res4', 'res5']
        if layer not in layers:
            raise ValueError('Layer {} does not exist.'.format(layer))

        x = concat_examples([prepare(img) for img in images])

        with chainer.function.no_backprop_mode(), chainer.using_config('train', False):
            x = Variable(self.xp.asarray(x))

            y = self(x, layers=[layer])[layer]
        return y

    def feature_vector(self, images):
        return self._layer_out(images, 'res5')


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.data.shape
    h = average_pooling_2d(x, (rows, cols), stride=1)
    h = np.reshape(h, (n, channel))
    return h
