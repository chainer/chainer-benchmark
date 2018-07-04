import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends
from benchmarks.utils import parameterize


@backends('gpu', 'cpu')
@parameterize([('batches', [1, 16])])
class Shift(FunctionBenchmark):
    def setup(self, batches):
        xp = self.xp

        # Prepare test data.
        n_channels = 32
        in_size = (128, 128)
        filter_size = (3, 3)
        dilate = 4

        xs_shape = (batches, n_channels) + in_size
        gy_shape = xs_shape

        xs = xp.random.uniform(-1, 1, xs_shape).astype(xp.float32)
        gy = xp.random.uniform(-1, 1, gy_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.shift, (xs, filter_size, dilate), gy)

    def time_forward(self, batches):
        self.forward()

    def time_backward(self, batches):
        self.backward()
