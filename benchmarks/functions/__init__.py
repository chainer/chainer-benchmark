import functools
import operator

import numpy

import chainer

from benchmarks import BenchmarkBase


class FunctionBenchmark(BenchmarkBase):

    """The base class for benchmark of functions."""

    # Call `test_*` methods only once as `backward()` has a side-effect.
    number = 1

    # Repeat the test for 10 times instead of 3 (`timeit.default_repeat`).
    repeat = 10

    def _convert_to_variable(self, x):
        """Maps each ndarray to a chainer.Variable.
        """
        if x is None:
            return None
        elif isinstance(x, (int, float, bool, str)):
            return x
        elif isinstance(x, (list, tuple)):
            return ([self._convert_to_variable(elem) for elem in x])
        else:
            return chainer.Variable(x)


    def _normalize_outputs(self, x):
        """Flattens outputs into a single tuple.
        """
        def rec(x):
            if isinstance(x, (list, tuple)):
                ret = []
                for elem in x:
                    ret.extend(rec(elem))
                return ret
            else:
                return [x]
        return None if x is None else tuple(rec(x))


    def setup_benchmark(self, function, inputs, grad_outputs=None):
        """Performs setup of benchmark for functions.

        Call this in `setup` method of your benchmark class.
        Note that this function performs forward computation.
        """
        self.function = function

        # Prepare for forward.

        self.forward_inputs = ([self._convert_to_variable(x) for x in inputs])

        # Prepare for backward.
        outputs = chainer.functions.identity(
            *self._normalize_outputs(self.forward()))

        if isinstance(outputs, (list, tuple)):
            self.forward_outputs = outputs
        else:
            self.forward_outputs = outputs,

        if grad_outputs is not None:
            grad_outputs = self._normalize_outputs(grad_outputs)
            assert len(grad_outputs) == len(self.forward_outputs)
            for i in range(len(grad_outputs)):
                self.forward_outputs[i].grad = grad_outputs[i]

    def forward(self):
        """Runs forward computation."""
        return self.function(*self.forward_inputs)

    def backward(self):
        """Runs backward computation."""
        self.forward_outputs[0].backward()


class UnaryMathFunctionBenchmark(FunctionBenchmark):

    """The base class for benchmark of unary element-wise math functions.

    Unlike `FunctionBenchmark`, this class automatically generates inputs and
    grads.
    """

    def setup_benchmark(
            self, function, shape=(1000, 1000), dtype=numpy.float32):
        inputs = (self.xp.arange(
            functools.reduce(operator.mul, shape),
            dtype=dtype).reshape(shape) + 1,)
        grad_outputs = (self.xp.array(inputs[0]),)
        super(UnaryMathFunctionBenchmark, self).setup_benchmark(
            function, inputs, grad_outputs)
