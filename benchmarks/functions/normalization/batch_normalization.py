import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends
from benchmarks.utils import parameterize


@backends('gpu', 'cpu')
@parameterize([('batches', [2, 16])])
class BatchNormalization(FunctionBenchmark):
    def setup(self, batches):
        xp = self.xp

        # Prepare test data.
        param_shape = (64, 64)
        ndim = 2
        shape = (batches,) + param_shape + (8,) * ndim

        gamma = xp.random.uniform(.5, 1, param_shape).astype(xp.float32)
        beta = xp.random.uniform(-1, 1, param_shape).astype(xp.float32)
        x = xp.random.uniform(-1, 1, shape).astype(xp.float32)
        gy = xp.random.uniform(-1, 1, shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.batch_normalization, (x, gamma, beta), gy)

    def time_forward(self, batches):
        self.forward()

    def time_backward(self, batches):
        self.backward()
