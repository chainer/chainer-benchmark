import chainer
from chainer import training

from benchmarks import BenchmarkBase
from benchmarks.utils import backends
from benchmarks.utils import is_backend_gpu
from benchmarks.utils import is_backend_ideep

from benchmarks.seq2seq.seq2seq import Seq2seq


@backends('gpu', 'gpu-cudnn')
class Benchmark(BenchmarkBase):

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

    def __init__(self, unit=1024, layer=3, dummy_data=None):
        # Setup dummy data generator
        if dummy_data is None:
            dummy_data = {}
        generator = DummyDataGenerator(**dummy_data)
        vocab_size = generator.get_vocab_size()

        # Setup model
        self.model = Seq2seq(layer, vocab_size, vocab_size, unit)
        self.gpu = -1
        if is_backend_gpu():
            self.gpu = 0
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.model.to_gpu()
        elif is_backend_ideep():
            self.model.to_intel64()

        # Generate dummy data for training and validation
        xp = self.model.xp
        self.train_data = generator.generate_training_data(xp)
        self.validate_data = generator.generate_validation_data(xp)

    def train(self, batchsize=64, epoch=3):
        # Setup optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(self.model)

        # Setup iterator
        train_iter = chainer.iterators.SerialIterator(
            self.train_data, batchsize)

        # Setup updater and trainer
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer, converter=_convert, device=self.gpu)
        trainer = training.Trainer(updater, (epoch, 'epoch'))

        # Add extensions (for debugging)
        trainer.extend(training.extensions.LogReport(
            trigger=(1, 'epoch')))
        trainer.extend(training.extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'main/perp', 'elapsed_time']),
            trigger=(1, 'epoch'))

        # Start training
        trainer.run()

    def translate(self):
        self.model.translate(self.validate_data)
