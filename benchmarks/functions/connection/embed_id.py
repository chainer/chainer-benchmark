import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends
from benchmarks.utils import parameterize


@backends('gpu', 'cpu')
@parameterize([('num_of_ids', [1, 128])])
class EmbedId(FunctionBenchmark):
    def setup(self, num_of_ids):
        xp = self.xp

        # Prepare test data.
        W_shape = (192, 256)
        y_shape = (num_of_ids, W_shape[1])

        x = xp.random.uniform(-1, 1, num_of_ids).astype(xp.int32)
        W = xp.random.uniform(-1, 1, W_shape).astype(xp.float32)
        y = xp.random.uniform(-1, 1, y_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.embed_id, (x, W), y)

    def time_forward(self, num_of_ids):
        self.forward()

    def time_backward(self, num_of_ids):
        self.backward()
