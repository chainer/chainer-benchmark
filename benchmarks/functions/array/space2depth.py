import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'cpu')
class Space2Depth(FunctionBenchmark):

    def setup(self):
        xp = self.xp
        x = xp.random.randn(128, 20, 60, 40).astype(xp.float32)
        gy = xp.random.randn(128, 80, 30, 20).astype(xp.float32)
        r = 2
        func = lambda x: F.space2depth(x, r)
        self.setup_benchmark(func, (x,), gy)

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
