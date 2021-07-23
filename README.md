# lightkde

[![Documentation Status](https://readthedocs.org/projects/lightkde/badge/?version=stable)](https://lightkde.readthedocs.io/en/stable/)
[![Continuous integration](https://github.com/rozsasarpi/lightkde/actions/workflows/push.yaml/badge.svg)](https://github.com/rozsasarpi/lightkde/actions)
[![PyPI version](https://img.shields.io/pypi/v/lightkde)](https://pypi.org/project/lightkde/)
![python versions](https://img.shields.io/pypi/pyversions/lightkde)
[![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rozsasarpi/bafe6e5b1382e4c2c49156a01e4803f3/raw/lightkde_main_coverage.json)](https://en.wikipedia.org/wiki/Code_coverage)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/rozsasarpi/lightkde.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rozsasarpi/lightkde/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/rozsasarpi/lightkde.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rozsasarpi/lightkde/alerts/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


A lightning fast, lightweight, and reliable kernel density estimation.

* Easy to use, e.g. ``density_vec, x_vec = kde_1d(sample_vec=sample)``\.
* Works with 1d and 2d samples.
* Works with weighted samples as well.
* Based on the MATLAB implementations of Botev:
  [kde](https://www.mathworks.com/matlabcentral/fileexchange/14034-kernel-density-estimator),
  [kde2d](https://www.mathworks.com/matlabcentral/fileexchange/17204-kernel-density-estimation).

![alt text](https://gist.githubusercontent.com/rozsasarpi/022fa396c919fbedabcd78fde9d1801a/raw/9822c2d457fcd5a7ef9b06350f14c9f16ae80b71/illustrative_image.svg)


## Install

```bash
pip install lightkde
```

## Usage

```python
import numpy as np
from lightkde import kde_1d

sample = np.random.randn(1000)

density_vec, x_vec = kde_1d(sample_vec=sample)
```

For further examples see the [documentation](https://lightkde.readthedocs.io/en/latest).

## Other kde packages

Other python packages for kernel density estimation:

* [scipy.stats.gaussian_kde](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
* [KDEpy](https://github.com/tommyod/KDEpy)
