import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'gpu-cudnn', 'cpu', 'cpu-ideep')
class NStepGRU(FunctionBenchmark):
    def setup(self):
        xp = self.xp

        # Prepare test data.
        batches = [8, 4, 2]
        n_layers = 3
        in_size = 24
        out_size = 16
        dropout = 0.0

        hx_shape = (n_layers, batches[0], out_size)

        xs = [xp.random.uniform(-1, 1, (b, in_size)).astype(xp.float32)
              for b in batches]
        hx = xp.random.uniform(-1, 1, hx_shape).astype(xp.float32)

        ws = []
        bs = []
        for i in range(n_layers):
            weights = []
            biases = []
            for j in range(6):
                if i == 0 and j < 3:
                    w_in = in_size
                else:
                    w_in = out_size

                weights.append(xp.random.uniform(
                    -1, 1, (out_size, w_in)).astype(xp.float32))
                biases.append(xp.random.uniform(
                    -1, 1, (out_size,)).astype(xp.float32))
            ws.append(weights)
            bs.append(biases)

        dhy = xp.random.uniform(-1, 1, hx_shape).astype(xp.float32)
        dys = [xp.random.uniform(-1, 1, (b, out_size)).astype(xp.float32)
               for b in batches]

        # Setup benchmark.
        self.setup_benchmark(F.n_step_gru,
                             (n_layers, dropout, hx, ws, bs, xs),
                             (dhy, dys))

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
