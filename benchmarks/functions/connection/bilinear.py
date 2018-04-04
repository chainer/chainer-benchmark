import numpy

import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'gpu-cudnn', 'cpu', 'cpu-ideep')
class Bilinear(FunctionBenchmark):
    def setup(self):
        xp = self.xp

        def uniform(*shape):
            return xp.random.uniform(-1, 1, shape).astype(numpy.float32)

        # Prepare test data.
        batches = 16
        e1_shape = (batches, 16)
        e2_shape = (batches, 20)
        e1_size = numpy.prod(e1_shape[1])
        e2_size = numpy.prod(e2_shape[1])
        out_size = 24

        e1 = uniform(*e1_shape)
        e2 = uniform(*e2_shape)
        W = uniform(e1_size, e2_size, out_size)
        V1 = uniform(e1_size, out_size)
        V2 = uniform(e2_size, out_size)
        b = uniform(out_size)
        gy = uniform(batches, out_size)

        # Setup benchmark.
        self.setup_benchmark(F.bilinear, (e1, e2, W, V1, V2, b), gy)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
