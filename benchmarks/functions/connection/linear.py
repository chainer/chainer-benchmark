import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'cpu', 'cpu-ideep')
class Linear(FunctionBenchmark):
    def setup(self):
        xp = self.xp

        # Prepare test data.
        x_shape = (128, 192)
        b_size = 256
        W_shape = (b_size, x_shape[1])
        y_shape = (x_shape[0], b_size)

        x = xp.random.uniform(-1, 1, x_shape).astype(xp.float32)
        W = xp.random.uniform(-1, 1, W_shape).astype(xp.float32)
        b = xp.random.uniform(-1, 1, b_size).astype(xp.float32)
        y = xp.random.uniform(-1, 1, y_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.linear, (x, W, b), y)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
