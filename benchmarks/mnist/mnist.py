import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from benchmarks import BenchmarkBase
from benchmarks.utils import backends
from benchmarks.utils import is_backend_gpu
from benchmarks.utils import is_backend_ideep
from benchmarks.utils import parameterize


class Network(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(Network, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class Application(object):

    def __init__(self, units):
        self.model = L.Classifier(Network(units, 10))

        self.gpu = -1
        if is_backend_gpu():
            self.gpu = 0
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()
        elif is_backend_ideep():
            self.model.to_intel64()

    def train(self, train, test, epoch, batchsize):
        optimizer = chainer.optimizers.MomentumSGD()
        optimizer.setup(self.model)

        train_iter = chainer.iterators.SerialIterator(train, batchsize)
        test_iter = chainer.iterators.SerialIterator(
            test, batchsize, repeat=False, shuffle=False)

        updater = training.updater.StandardUpdater(
            train_iter, optimizer, device=self.gpu)
        trainer = training.Trainer(updater, (epoch, 'epoch'))
        trainer.extend(extensions.Evaluator(
            test_iter, self.model, device=self.gpu))

        trainer.run()

    def predict(self, data):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            self.model.predictor(data)


class _MLPBase(BenchmarkBase):
    def setup(self, units, batchsize):
        self.app = Application(units=units)
        self.train_data, self.test_data = chainer.datasets.get_mnist()

        xp = self.xp
        arrays = [xp.asarray(x[0]) for x in self.train_data]
        self.predict_data = xp.concatenate(arrays).reshape(len(arrays), -1)

        # Initialize.
        self.app.predict(self.predict_data[0:1])


@backends('gpu', 'gpu-cudnn', 'cpu', 'cpu-ideep')
@parameterize([
    ('units', [10, 100, 150]),
    ('batchsize', [100]),
])
class MLPTrain(_MLPBase):
    def time_train(self, units, batchsize):
        self.app.train(
            self.train_data, self.test_data, epoch=1, batchsize=batchsize)


@backends('gpu', 'gpu-cudnn', 'cpu', 'cpu-ideep')
@parameterize([
    ('units', [10, 100, 150, 1000]),
    ('batchsize', [1, 100]),
])
class MLPPredict(_MLPBase):
    def time_predict(self, units, batchsize):
        self.app.predict(chainer.Variable(self.predict_data[0:batchsize]))
