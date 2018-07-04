import functools
import operator

import numpy

import chainer.functions as F
from chainer.backends import intel64

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends
from benchmarks.utils import is_backend_ideep
from benchmarks.utils import parameterize


@backends('gpu', 'cpu', 'cpu-ideep')
@parameterize([('count', [2, 4, 16])])
class AddFunc(FunctionBenchmark):

    def setup(self, count):
        shape = (5000, 5000)
        dtype = numpy.float32
        size = functools.reduce(operator.mul, shape)
        inputs = [
            self.xp.arange(size, dtype=dtype).reshape(shape) + 1
            for _ in range(count)
        ]
        grad_outputs = self.xp.ones(shape, dtype=dtype)

        if is_backend_ideep():
            inputs = [intel64.ideep.array(x) for x in inputs]
            grad_outputs = intel64.ideep.array(grad_outputs)

        self.setup_benchmark(F.add, inputs, grad_outputs)

    def time_forward(self, count):
        self.forward()

    def time_backward(self, count):
        self.backward()
