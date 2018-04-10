import numpy

import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'gpu-cudnn', 'cpu')
class DeconvolutionND(FunctionBenchmark):
    def setup(self):
        xp = self.xp

        # Prepare test data.
        batches = 128
        in_channels = 16
        out_channels = 3
        in_size = (9, 9, 9)
        filter_size = (8, 8, 8)

        out_size = tuple(map((lambda x, k: x - 1 + k), in_size, filter_size))

        W_scale = xp.sqrt(1. / (numpy.prod(filter_size) * in_channels))
        in_shape = (batches, in_channels) + in_size
        W_shape = (in_channels, out_channels) + filter_size
        out_shape = (batches, out_channels) + out_size

        x = xp.random.uniform(-1, 1, in_shape).astype(xp.float32)
        W = xp.random.normal(0, W_scale, W_shape).astype(xp.float32)
        b = xp.random.uniform(-1, 1, out_channels).astype(xp.float32)
        gy = xp.random.uniform(-1, 1, out_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.deconvolution_nd, (x, W, b), gy)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
