# Utilities (Chainer-specific)
from benchmarks.utils.backend import backends  # NOQA
from benchmarks.utils.backend import have_ideep  # NOQA
from benchmarks.utils.backend import is_backend_gpu  # NOQA
from benchmarks.utils.backend import is_backend_ideep  # NOQA

from benchmarks.utils.config import config  # NOQA

# Utilities (available for both Chainer and CuPy)
from benchmarks.utils.helper import parameterize  # NOQA
from benchmarks.utils.helper import sync  # NOQA
