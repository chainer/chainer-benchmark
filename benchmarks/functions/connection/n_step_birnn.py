import chainer.functions as F

from benchmarks.functions import FunctionBenchmark
from benchmarks.utils import backends


@backends('gpu', 'gpu-cudnn', 'cpu', 'cpu-ideep')
class NStepBiRNN(FunctionBenchmark):
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
        h_shape = (n_layers * 2, batches[0], out_size)
        hx = _shaped_random(h_shape)

        i = in_size
        o = out_size
        ws = []
        bs = []
        # First layer has the different shape
        for di in range(2):
            ws.append(_shaped_random([(o, i), (o, o)]))
            bs.append(_shaped_random([o, o]))
        # Rest layers
        for _ in range(n_layers - 1):
            for di in range(2):
                ws.append(_shaped_random([(o, o * 2), (o, o)]))
                bs.append(_shaped_random([o, o]))

        dys = _shaped_random(
            [(b, out_size * 2) for b in batches])
        dhy = _shaped_random(h_shape)

        # Setup benchmark.
        self.setup_benchmark(F.n_step_birnn,
                             (n_layers, dropout, hx, ws, bs, xs),
                             (dhy, dys))

    def time_forward(self):
        self.forward()

    def time_backward(self):
        self.backward()
