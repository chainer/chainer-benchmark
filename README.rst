Chainer Benchmark
=================

Benchmarking Chainer with Airspeed Velocity.

Requirements
------------

You need to install ``asv`` benchmark framework and ``Cython`` (to build CuPy).

.. code-block:: console

    pip install asv Cython

The benchmark can be run over Chainer v3.1.0+ or v4.0.0b1+.

Usage
-----

.. code-block:: sh

    # Enable ccache for performance (optional).
    export PATH="/usr/lib/ccache:${PATH}"
    export NVCC="ccache nvcc"

    # Run benchmark against target commit-ish of Chainer and CuPy.
    # Note that specified versions must be a compatible combination.
    # You can use `find_cupy_version.py` helper tool to get appropriate CuPy
    # version/commit for the given Chainer version/commit.
    ./run.sh master master
    ./run.sh v4.0.0b4 v4.0.0b4

    # Compare the benchmark results between two commits to see regression
    # and/or performance improvements in command line.
    alias git_commit='git show --format="%H"'
    asv compare $(git_commit v4.0.0b4) $(git_commit master)

    # Convert the results into HTML.
    # The result will be in `html` directory.
    asv publish

    # Start the HTTP server to browse HTML.
    asv preview

Alternatively you can use Docker.

.. code-block:: sh

    # Build docker image for benchmark.
    docker build -t chainer-benchmark docker

    # Create a machine configuration file (`.asv-machine.json`) in this directory (first time only).
    nvidia-docker run --rm -u $(id -u):$(id -g) -v ${PWD}:/benchmarks -w /benchmarks -e HOME=/benchmarks chainer-benchmark asv machine --machine $(hostname)

    # Run benchmark.
    nvidia-docker run --rm -u $(id -u):$(id -g) -v ${PWD}:/benchmarks -w /benchmarks -e HOME=/benchmarks chainer-benchmark ./run.sh master master --machine $(hostname)
