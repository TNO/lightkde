""""
Some tests compare the results against the results from Botev's Matlab implementation:
 * `kde.m`: https://mathworks.com/matlabcentral/fileexchange/14034-kernel-density-estimator  # noqa E501

np.testing.assert_allclose
atol + rtol * abs(desired)

TODO:
    * changing the mesh generation would break this code -> interpolation would
        solve the potential problem of the mismatch of meshes, to be added later.
    * extend tests to cover branches.
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, norm

from lightkde import kde_1d
from tests import TEST_DATA_BASE_URL

plot_fig = False
LOGGER = logging.getLogger(__name__)


def test_kde_1d_gaussian_mixture_against_matlab():
    sample = np.loadtxt(
        fname=os.path.join(
            TEST_DATA_BASE_URL, "1d_sample%23gaussian_mixture%23matlab.txt"
        )
    )
    expected_density_vec = np.loadtxt(
        fname=os.path.join(
            TEST_DATA_BASE_URL, "1d_density%23gaussian_mixture%23matlab.txt"
        )
    )
    expected_x_vec = np.loadtxt(
        fname=os.path.join(
            TEST_DATA_BASE_URL, "1d_x_mesh%23gaussian_mixture%23matlab.txt"
        )
    )

    density_vec, x_vec = kde_1d(sample_vec=sample, n_x_vec=2 ** 14, x_min=-15, x_max=15)

    np.testing.assert_allclose(x_vec, expected_x_vec, atol=1e-3)
    np.testing.assert_allclose(density_vec, expected_density_vec, atol=5e-3)

    if plot_fig is True:
        plt.subplots()
        plt.plot(expected_x_vec, expected_density_vec, label="expected (Matlab)")
        plt.plot(x_vec, density_vec, "--r", label="lightkde.py")
        plt.legend()
        plt.show()


def test_kde_1d_gaussian_against_scipy():
    """
    We compare only against a unimodal distribution as "bimodal or multi-modal
    distributions tend to be oversmoothed" by `scipy.stats.gaussian_kde`, i.e. it is
    not a reliable reference solution.
    """
    np.random.seed(42)
    sample = norm.rvs(size=1000)

    # return bandwidth to go through that part of the code as well, asserting the
    # density verifies the bandwidth too
    density_vec, x_vec, bandwidth = kde_1d(sample_vec=sample, return_bandwidth=True)

    # scipy
    gkde = gaussian_kde(dataset=sample)
    expected_density_vec = gkde.evaluate(x_vec)

    np.testing.assert_allclose(density_vec, expected_density_vec, atol=1e-2)

    if plot_fig is True:
        plt.subplots()
        plt.plot(x_vec, expected_density_vec, label="scipy.stats.gaussian_kde")
        plt.plot(x_vec, density_vec, "--r", label="lightkde")
        plt.legend()
        plt.show()


def test_kde_1d_gaussian_with_weights_against_scipy():
    """
    We compare only against a unimodal distribution as "bimodal or multi-modal
    distributions tend to be oversmoothed" by `scipy.stats.gaussian_kde`, i.e. it is
    not a reliable reference solution.
    """
    np.random.seed(42)
    n_sample = 2000
    sample = norm.rvs(size=n_sample)
    weights = np.ones(n_sample)
    weights[0 : int(n_sample / 3)] = 0.5
    sample = np.sort(sample)

    density_vec, x_vec = kde_1d(
        sample_vec=sample, n_x_vec=2 ** 14, x_min=-5, x_max=5, weight_vec=weights
    )
    density_vec_nw, x_vec_nw = kde_1d(
        sample_vec=sample, n_x_vec=2 ** 14, x_min=-5, x_max=5
    )
    # scipy
    gkde = gaussian_kde(dataset=sample, weights=weights)
    expected_density_vec = gkde.evaluate(x_vec)

    np.testing.assert_allclose(density_vec, expected_density_vec, atol=1e-2)

    if plot_fig is True:
        plt.subplots()
        plt.plot(x_vec, expected_density_vec, label="scipy.stats.gaussian_kde")
        plt.plot(x_vec, density_vec, "--r", label="lightkde")
        plt.plot(x_vec_nw, density_vec_nw, "--g", label="lightkde; no weight")
        plt.legend()
        plt.show()


def test_kde_1d_illustrative_image():
    np.random.seed(42)
    sample = np.hstack((norm.rvs(size=2_000), 0.3 * norm.rvs(size=1_000) + 5))

    density_vec, x_vec = kde_1d(sample_vec=sample)

    gkde = gaussian_kde(dataset=sample)
    scipy_density_vec = gkde.evaluate(x_vec)

    plt.plot(x_vec, density_vec, "--r", label="lightkde")
    plt.plot(x_vec, scipy_density_vec, label="scipy.stats.gaussian_kde")
    plt.hist(sample, bins=100, density=True, alpha=0.5, label="data")
    plt.legend()
    plt.tight_layout()
    plt.axis("off")

    if plot_fig is True:
        plt.savefig("illustrative_image.svg")


def test_kde_1d_no_boundaries():
    np.random.seed(42)
    sample = np.random.randn(1000)

    density_vec_b, x_vec_b = kde_1d(sample_vec=sample, x_min=-5, x_max=5)
    density_vec_nb, x_vec_nb = kde_1d(sample_vec=sample)

    interp_fun = interp1d(x_vec_b, density_vec_b)
    density_vec_nb_interp = interp_fun(x_vec_nb)

    np.testing.assert_allclose(density_vec_nb_interp, density_vec_nb, atol=1e-4)


def test_kde_1d_scipy_fallback(caplog):
    # https://stackoverflow.com/a/60522543/4063376
    sample = np.random.random(5)
    caplog.set_level(logging.WARNING)
    _, _ = kde_1d(sample_vec=sample)
    assert "Failed to find the optimal bandwidth" in caplog.text

    caplog.clear()
    _, _, _ = kde_1d(sample_vec=sample, return_bandwidth=True)
    assert "Failed to find the optimal bandwidth" in caplog.text


def test_kde_1d_wrong_weights_input():
    with pytest.raises(ValueError):
        sample = np.random.random(5)
        kde_1d(sample_vec=sample, weight_vec=[1, 2])
    with pytest.raises(ValueError):
        sample = np.random.random(3)
        kde_1d(sample_vec=sample, weight_vec=[-1, 2, 3])


def test_kde_1d_wrong_output_number():
    with pytest.raises(ValueError, match="too many values to unpack"):
        sample = norm.rvs(size=100)
        _, _ = kde_1d(sample_vec=sample, return_bandwidth=True)

    with pytest.raises(ValueError, match="not enough values to unpack"):
        sample = norm.rvs(size=100)
        _, _, _ = kde_1d(sample_vec=sample, return_bandwidth=False)
