import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends
from benchmarks.utils import parameterize


@backends('gpu', 'gpu-cudnn', 'cpu', 'cpu-ideep')
@parameterize([('batches', [1, 16])])
class Convolution2D(FunctionBenchmark):
    def setup(self, batches):
        xp = self.xp

        # Prepare test data.
        in_channels = 3
        out_channels = 64
        ih, iw = (128, 128)
        kh, kw = (12, 12)
        x = xp.random.uniform(
            -1, 1, (batches, in_channels, ih, iw)).astype(xp.float32)
        W = xp.random.normal(
            0, xp.sqrt(1. / (kh * kw * in_channels)),
            (out_channels, in_channels, kh, kw)).astype(xp.float32)
        b = xp.random.uniform(-1, 1, out_channels).astype(xp.float32)
        gy = xp.random.uniform(
            -1, 1, (batches, out_channels, 117, 117)).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.convolution_2d, (x, W, b), gy)

    def time_forward(self, batches):
        self.forward()

    def time_backward(self, batches):
        self.backward()
