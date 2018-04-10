import numpy

import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'cpu')
class DepthwiseConvolution2D(FunctionBenchmark):
    def setup(self):
        xp = self.xp

        # Prepare test data.
        batches = 32
        in_channels = 8
        channel_multiplier = 2
        in_size = (128, 128)
        filter_size = (12, 12)

        out_channels = in_channels * channel_multiplier
        out_size = tuple(map((lambda x, k: x - k + 1), in_size, filter_size))

        x_shape = (batches, in_channels) + in_size
        W_scale = xp.sqrt(1. / (numpy.prod(filter_size) * in_channels))
        W_shape = (channel_multiplier, in_channels) + filter_size
        b_shape = out_channels
        gy_shape = (batches, out_channels) + out_size

        x = xp.random.uniform(-1, 1, x_shape).astype(xp.float32)
        W = xp.random.normal(0, W_scale, W_shape).astype(xp.float32)
        b = xp.random.uniform(-1, 1, b_shape).astype(xp.float32)
        gy = xp.random.uniform(-1, 1, gy_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.depthwise_convolution_2d, (x, W, b), gy)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
