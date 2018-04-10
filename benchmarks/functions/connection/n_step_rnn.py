import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'gpu-cudnn', 'cpu')
class NStepRNN(FunctionBenchmark):
    def setup(self):
        xp = self.xp

        def _shaped_random(shape):
            if isinstance(shape, list):
                return [_shaped_random(s) for s in shape]
            else:
                return xp.random.uniform(-1, 1, shape).astype(xp.float32)

        # Prepare test data.
        batches = [8, 4, 2]
        n_layers = 3
        in_size = 24
        out_size = 16
        dropout = 0.0

        xs = _shaped_random([(b, in_size) for b in batches])
        hx_shape = (n_layers, batches[0], out_size)
        hx = _shaped_random(hx_shape)

        o = out_size
        i = in_size
        ws = []
        bs = []
        # The first layer has the different shape
        ws.append(_shaped_random([(o, i), (o, o)]))
        bs.append(_shaped_random([o, o]))
        for _ in range(n_layers - 1):
            ws.append(_shaped_random([(o, o), (o, o)]))
            bs.append(_shaped_random([o, o]))

        dys = _shaped_random([(b, out_size) for b in batches])
        dhy = _shaped_random(hx_shape)

        hx_shape = (n_layers, batches[0], out_size)

        xs = [xp.random.uniform(-1, 1, (b, in_size)).astype(xp.float32)
              for b in batches]
        hx = xp.random.uniform(-1, 1, hx_shape).astype(xp.float32)

        # Setup benchmark.
        self.setup_benchmark(F.n_step_rnn,
                             (n_layers, dropout, hx, ws, bs, xs),
                             (dhy, dys))

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
