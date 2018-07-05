import chainer
from chainer import training

from benchmarks import BenchmarkBase
from benchmarks.utils import backends
from benchmarks.utils import is_backend_gpu
from benchmarks.utils import is_backend_ideep

from benchmarks.seq2seq.seq2seq import Seq2seq


@backends('gpu', 'gpu-cudnn')
class Benchmark(BenchmarkBase):

    number = 1

    def setup(self):
        self._app = Application()

    def time_training(self):
        self._app.train()

    def time_translate(self):
        self._app.translate()


class DummyDataGenerator(object):

    """Generates a dummy data for Seq2seq example."""

    def __init__(self, count=2048, vocab_count=40000, words_per_data=10):
        self._count = count

        # +2 for UNK and EOS
        self._vocab_size = vocab_count + 2
        self._words_per_data = words_per_data
        assert self._words_per_data <= self._vocab_size

    def get_vocab_size(self):
        """Returns the number of vocabularies, including UNK and EOS."""

        return self._vocab_size

    def _generate_sentence(self, xp):
        """Returns the dummy sentence.

        The returned value is a single ``xp.ndarray``.
        """

        # +2 for UNK and EOS
        sentence = xp.arange(self._words_per_data, dtype=xp.int32) + 2
        xp.random.shuffle(sentence)
        return sentence

    def generate_training_data(self, xp):
        """Returns the dummy data for training.

        The returned value is a list of instances.
        A instance is a tuple of two ``xp.ndarray``\\ s which represents
        a set of source sentence and target sentence.
        """

        return [(self._generate_sentence(xp), self._generate_sentence(xp))
                for _ in range(self._count)]

    def generate_validation_data(self, xp):
        """Returns the dummy data for validation.

        The returned value is a list of ``xp.ndarray``, which represents
        a input sentence.
        """

        return [self._generate_sentence(xp) for _ in range(self._count)]


def _convert(batch, device):
    # Arrays in the batch are already on the target device.
    return {'xs': [x for x, _ in batch], 'ys': [y for _, y in batch]}


class Application(object):

    def __init__(self, unit=1024, layer=3, train_batchsize=64, train_epoch=3,
                 dummy_data=None):
        # Setup dummy data generator
        if dummy_data is None:
            dummy_data = {}
        generator = DummyDataGenerator(**dummy_data)
        vocab_size = generator.get_vocab_size()

        # Setup model
        model = Seq2seq(layer, vocab_size, vocab_size, unit)
        gpu = -1
        if is_backend_gpu():
            gpu = 0
            chainer.cuda.get_device_from_id(gpu).use()
            model.to_gpu()
        elif is_backend_ideep():
            model.to_intel64()

        # Generate dummy data for training and validation
        xp = model.xp
        train_data = generator.generate_training_data(xp)
        validate_data = generator.generate_validation_data(xp)

        # Setup optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        # Setup iterator
        train_iter = chainer.iterators.SerialIterator(
            train_data, train_batchsize)

        # Setup updater and trainer
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer, converter=_convert, device=gpu)
        trainer = training.Trainer(updater, (train_epoch, 'epoch'))

        # Add extensions (for debugging)
        trainer.extend(training.extensions.LogReport(
            trigger=(1, 'epoch')))
        trainer.extend(training.extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'main/perp', 'elapsed_time']),
            trigger=(1, 'epoch'))

        # Variables to be used in the benchmark.
        self.trainer = trainer
        self.model = model
        self.validate_data = validate_data

    def train(self):
        # Start training
        self.trainer.run()

    def translate(self):
        self.model.translate(self.validate_data)
