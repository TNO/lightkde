""""
Some tests compare the results against the results from Botev's Matlab implementation:
 * `kde2d.m`: https://mathworks.com/matlabcentral/fileexchange/17204-kernel-density-estimation  # noqa E501

np.testing.assert_allclose
atol + rtol * abs(desired)

TODO:
    * changing the mesh generation would break this code -> interpolation would
        solve the potential problem of the mismatch of meshes, to be added later.
    * extend tests to cover branches.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import gaussian_kde, norm

from lightkde.lightkde import dct2d, hist_2d, idct2d, kde_2d
from tests import TEST_DATA_BASE_URL

plot_fig = False


def test_kde_2d_simple_against_matlab():
    x = [0.11, 0.21, 0.31, 0.31, 0.21]
    y = [0.61, 0.31, 0.91, 0.91, 0.31]
    sample = np.vstack((x, y)).T
    density_mx, x_mx, y_mx = kde_2d(
        sample_mx=sample, n_row_mx=4, xy_min=[0, 0], xy_max=[1, 1]
    )

    expected_x_mx = np.array(
        [
            [0, 3.333333333333333e-01, 6.666666666666667e-01, 1.000000000000000e00],
            [0, 3.333333333333333e-01, 6.666666666666667e-01, 1.000000000000000e00],
            [0, 3.333333333333333e-01, 6.666666666666667e-01, 1.000000000000000e00],
            [0, 3.333333333333333e-01, 6.666666666666667e-01, 1.000000000000000e00],
        ]
    )
    expected_y_mx = np.array(
        [
            [0, 0, 0, 0],
            [
                3.333333333333333e-01,
                3.333333333333333e-01,
                3.333333333333333e-01,
                3.333333333333333e-01,
            ],
            [
                6.666666666666667e-01,
                6.666666666666667e-01,
                6.666666666666667e-01,
                6.666666666666667e-01,
            ],
            [
                1.000000000000000e00,
                1.000000000000000e00,
                1.000000000000000e00,
                1.000000000000000e00,
            ],
        ]
    )
    expected_density_mx = np.array(
        [
            [
                1.656375795026443e00,
                6.405166075308758e-01,
                8.212670025790514e-02,
                4.601619765514396e-03,
            ],
            [
                2.319920561403144e00,
                1.067794861703838e00,
                2.156967165887120e-01,
                1.831421342813525e-02,
            ],
            [
                2.427512383180407e00,
                1.658453039663006e00,
                5.448587632946583e-01,
                5.679296900586699e-02,
            ],
            [
                2.093705214644155e00,
                2.192159811306726e00,
                9.192163046403170e-01,
                1.019544385602958e-01,
            ],
        ]
    )

    np.testing.assert_allclose(x_mx, expected_x_mx, atol=1e-3)
    np.testing.assert_allclose(y_mx, expected_y_mx, atol=1e-3)
    np.testing.assert_allclose(density_mx, expected_density_mx, atol=5e-3)

    if plot_fig is True:
        plt.subplots()
        plt.contour(expected_x_mx, expected_y_mx, expected_density_mx, 5)
        plt.contour(x_mx, y_mx, density_mx, 5, colors="red", linestyles="dotted")
        plt.axis("equal")
        plt.show()


def test_kde_2d_gaussian_mixture_against_matlab():
    sample = np.loadtxt(
        fname=os.path.join(
            TEST_DATA_BASE_URL, "2d_sample%23gaussian_mixture%23matlab.txt"
        ),
        delimiter=",",
    )
    expected_density_mx = np.loadtxt(
        fname=os.path.join(
            TEST_DATA_BASE_URL, "2d_density%23gaussian_mixture%23matlab.txt"
        ),
        delimiter=",",
    )
    expected_x_mx = np.loadtxt(
        fname=os.path.join(
            TEST_DATA_BASE_URL, "2d_x_mesh%23gaussian_mixture%23matlab.txt"
        ),
        delimiter=",",
    )
    expected_y_mx = np.loadtxt(
        fname=os.path.join(
            TEST_DATA_BASE_URL, "2d_y_mesh%23gaussian_mixture%23matlab.txt"
        ),
        delimiter=",",
    )

    density_mx, x_mx, y_mx = kde_2d(
        sample_mx=sample, n_row_mx=2 ** 8, xy_min=[-10, -5], xy_max=[10, 5]
    )

    np.testing.assert_allclose(x_mx, expected_x_mx, atol=1e-3)
    np.testing.assert_allclose(y_mx, expected_y_mx, atol=1e-3)
    np.testing.assert_allclose(density_mx, expected_density_mx, atol=5e-3)

    if plot_fig is True:
        plt.subplots()
        plt.contour(expected_x_mx, expected_y_mx, expected_density_mx, 5)
        plt.contour(x_mx, y_mx, density_mx, 5, colors="red", linestyles="dotted")
        plt.show()


def test_kde_2d_sinusoidal_against_matlab():
    sample = np.loadtxt(
        fname=os.path.join(TEST_DATA_BASE_URL, "2d_sample%23sinusoidal%23matlab.txt"),
        delimiter=",",
    )
    expected_density_mx = np.loadtxt(
        fname=os.path.join(TEST_DATA_BASE_URL, "2d_density%23sinusoidal%23matlab.txt"),
        delimiter=",",
    )
    expected_x_mx = np.loadtxt(
        fname=os.path.join(TEST_DATA_BASE_URL, "2d_x_mesh%23sinusoidal%23matlab.txt"),
        delimiter=",",
    )
    expected_y_mx = np.loadtxt(
        fname=os.path.join(TEST_DATA_BASE_URL, "2d_y_mesh%23sinusoidal%23matlab.txt"),
        delimiter=",",
    )

    density_mx, x_mx, y_mx = kde_2d(
        sample_mx=sample, n_row_mx=2 ** 8, xy_min=[0, -2], xy_max=[1, 2]
    )

    np.testing.assert_allclose(x_mx, expected_x_mx, atol=1e-3)
    np.testing.assert_allclose(y_mx, expected_y_mx, atol=1e-3)
    np.testing.assert_allclose(density_mx, expected_density_mx, atol=5e-3)

    if plot_fig is True:
        plt.subplots()
        plt.contour(expected_x_mx, expected_y_mx, expected_density_mx, 5)
        plt.contour(x_mx, y_mx, density_mx, 5, colors="red", linestyles="dotted")
        plt.show()


def test_kde_2d_gaussian_against_scipy():
    """
    We compare only against a unimodal distribution as "bimodal or multi-modal
    distributions tend to be oversmoothed" by `scipy.stats.gaussian_kde`, i.e. it is
    not a reliable reference solution.
    """
    np.random.seed(42)
    sample = norm.rvs(size=(2000, 2))

    # return bandwidth to go through that part of the code as well, asserting the
    # density verifies the bandwidth too
    density_mx, x_mx, y_mx, bandwidth = kde_2d(sample_mx=sample, return_bandwidth=True)

    # scipy
    gkde = gaussian_kde(dataset=sample.T)
    xy_mx = np.hstack((x_mx.reshape(-1, 1), y_mx.reshape(-1, 1)))
    expected_density_mx = gkde.evaluate(xy_mx.T).reshape(x_mx.shape)

    np.testing.assert_allclose(density_mx, expected_density_mx, atol=1e-2)

    if plot_fig is True:
        plt.subplots()
        plt.contour(x_mx, y_mx, expected_density_mx, 5)
        plt.contour(x_mx, y_mx, density_mx, 5, colors="red", linestyles="dotted")
        plt.show()


def test_kde_2d_gaussian_with_weights_against_scipy():
    """
    We compare only against a unimodal distribution as "bimodal or multi-modal
    distributions tend to be oversmoothed" by `scipy.stats.gaussian_kde`, i.e. it is
    not a reliable reference solution.
    """
    np.random.seed(42)
    n_sample = 3000
    sample = norm.rvs(size=(n_sample, 2))
    weights = np.ones(n_sample)
    weights[0 : int(n_sample / 3)] = 0.5
    sample = sample[sample[:, 1].argsort()]

    density, x_mesh, y_mesh = kde_2d(sample_mx=sample, weight_vec=weights)
    density_nw, x_mesh_nw, y_mesh_nw = kde_2d(sample_mx=sample)

    # scipy
    gkde = gaussian_kde(dataset=sample.T, weights=weights)
    xy_mesh = np.hstack((x_mesh.reshape(-1, 1), y_mesh.reshape(-1, 1)))
    expected_density = gkde.evaluate(xy_mesh.T).reshape(x_mesh.shape)

    np.testing.assert_allclose(density, expected_density, atol=1e-2)

    if plot_fig is True:
        plt.subplots()
        plt.contour(x_mesh, y_mesh, expected_density, 5, colors="blue")
        plt.contour(x_mesh, y_mesh, density, 5, colors="red", linestyles="dotted")
        plt.contour(
            x_mesh_nw, y_mesh_nw, density_nw, 5, colors="green", linestyles="dotted"
        )
        plt.show()


def test_kde_2d_wrong_sample_input():
    with pytest.raises(ValueError):
        sample = np.random.random((5, 3))
        kde_2d(sample_mx=sample)


def test_kde_2d_wrong_weights_input():
    with pytest.raises(ValueError):
        sample = np.random.random((5, 2))
        kde_2d(sample_mx=sample, weight_vec=[1, 2])
    with pytest.raises(ValueError):
        sample = np.random.random((3, 2))
        kde_2d(sample_mx=sample, weight_vec=[-1, 2, 3])


# --------------------------------------------------------------------------------------
# 2D utilities
# --------------------------------------------------------------------------------------
def test_dct2d_idct2d():
    sample = np.random.randn(16, 16)
    np.testing.assert_allclose(idct2d(dct2d(sample)), sample)


def test_hist_2d_against_matlab():
    """
    Testing for 2d sample_vec as that's the one only needed in kde2d
    reference solution: kde2d.m function binned_sample=hist_2d(sample_vec,M)
    """
    x = [0.11, 0.21, 0.31, 0.31, 0.21]
    y = [0.61, 0.31, 0.91, 0.91, 0.31]
    sample = np.vstack((x, y)).T
    m = 5

    expected_binned_sample = np.array(
        [
            [0.0, 0.0, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    binned_sample = hist_2d(sample, m)
    np.testing.assert_allclose(binned_sample, expected_binned_sample)
