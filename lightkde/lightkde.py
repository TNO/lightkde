"""
Reliable and extremely fast kernel density estimator for one and two-dimensional
samples.

The kernel density estimations here are kept as simple and as separated from the rest
of the code as possible. They do nothing but kernel density estimation. The
motivation for their partial reimplementation is that the existing kernel density
estimators are:
* suboptimal (like scipy where no kernel bandwidth optimization is done), or
* come with a gorilla holding a banana and the entire jungle although only the
    banana is needed.

Do one thing and do it well.

Botev's Matlab codes are the starting point of this implementation as those mostly
follow the above principle.

TODO:
 - [low] add cdf estimate as in ``kde_1d.m``.
 - [high] more thorough input check, mostly shape and type.
 - [high] check the details of ``histc`` in Matlab and ``np.histogram`` make sure that
    appending a zero to ``sample_hist`` is always valid.
"""

import copy
import logging
from typing import Iterable, Tuple, Union

import numpy as np
from scipy import fft, optimize
from scipy.stats import gaussian_kde

N_X_VEC = int(2 ** 14)
N_ROW_MX = int(2 ** 8)


# ======================================================================================
# 1D
# ======================================================================================
def kde_1d(
    sample_vec: Union[np.ndarray, list],
    n_x_vec: int = N_X_VEC,
    x_min: Union[int, float] = None,
    x_max: Union[int, float] = None,
    weight_vec: Union[np.ndarray, list] = None,
    return_bandwidth: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]:
    """
    Reliable and extremely fast kernel density estimator for one-dimensional sample.

    Gaussian kernel is assumed and the bandwidth is chosen automatically.
    Unlike many other implementations, this one is immune to problems caused by
    multimodal densities with widely separated modes. The estimation does not
    deteriorate for multimodal densities, because we never assume a parametric model
    for the sample.

    .. note::

        * The elements of ``sample_vec`` that fall between ``x_min`` and ``x_max`` will
          be treated as the full sample, i.e. the kernel density over ``[x_min, x_max]``
          will integrate to one.

        * If the search for finding the optimal bandwidth fails the functions falls
          back to ``scipy.stats.gaussian_kde``.

    Args:
        sample_vec:
            A vector of sample points from which the density estimate is constructed.
        n_x_vec:
            The number of ``x_vec`` points used in the uniform discretization of
            the interval ``[x_min, x_max]``. ``n_x_vec`` has to be a power of two. If
            ``n_x_vec`` is not a power of two, then ``n_x_vec`` is rounded up to the
            next power of two, i.e., ``n_x_vec`` is set to
            ``n_x_vec=2**ceil(log2(n_x_vec))``; the default value of ``n_x_vec`` is
            ``n_x_vec=2**14``.
        x_min:
            The lower boundary of the interval over which the density estimate is
            constructed.
        x_max:
            The upper boundary of the interval over which the density estimate is
            constructed.
        weight_vec:
            Weights of sample points. This must have the same shape as ``sample_vec``.
            If ``None`` (default), the samples are assumed to be equally weighted.
            Only the values of elements relative to each other matter,
            i.e. multiplying ``weight_vec`` by a non-negative scalar does not change
            the results.
        return_bandwidth:
            Should the used bandwidth be returned?

    Raises:
        ValueError: If ``weight_vec`` has at least one negative value.

    Warns:
        Root finding failed (Brent's method): Optimal bandwidth finding failed,
            falling back to the rule-of-thumb bandwidth of ``scipy.stats.gaussian_kde``.

    Returns:
        Kernel densities, a vector of length ``n_x_vec`` with the values of
        the density estimate at the grid points (``x_vec``).

        Kernel density grid (``x_vec``), a vector of grid points over which
        the kernel density estimate is computed.

        Optimal bandwidth (Gaussian kernel assumed), returned only if
        ``return_bandwidth`` is ``True``.


    Examples:
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            from lightkde import kde_1d

        .. code-block:: python

            sample_vec = [
                -1.3145, -0.5197, 0.9326, 3.2358, 0.3814,
                -0.3226, 2.1121, 1.1357, 0.4376, -0.0332
            ]
            density_vec, x_vec = kde_1d(sample_vec)

        .. code-block:: python

            sample_vec = np.hstack((np.random.normal(loc=-8, size=100),
                np.random.normal(loc=-3, size=100),
                np.random.normal(loc=7, size=100)))
            density_vec, x_vec = kde_1d(sample_vec)

            plt.subplots()
            plt.plot(x_vec, density_vec)
            plt.show()

    The kde bandwidth selection method is outlined in [1]. This implementation is
    based on the implementation of Daniel B. Smith [2] who based his
    implementation on the Matlab implementation by Zdravko Botev [3].

    References:
        [1] Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010) Annals of
        Statistics, Volume 38, Number 5, pages 2916-2957.

        [2] https://github.com/Daniel-B-Smith/KDE-for-SciPy/blob/a9982909bbb92a7e243e5fc9a74f957d883f1c5d/kde.py # noqa: E501
        Updated on: 6 Feb 2013.

        [3] https://nl.mathworks.com/matlabcentral/fileexchange/14034-kernel-density-estimator # noqa: E501
        Updated on: 30 Dec 2015.
    """
    sample_vec = np.array(sample_vec).ravel()
    n_sample = len(np.unique(sample_vec))

    # Parameters to set up the x_vec on which to calculate
    n_x_vec = int(2 ** np.ceil(np.log2(n_x_vec)))
    if x_min is None or x_max is None:
        sample_min = np.min(sample_vec)
        sample_max = np.max(sample_vec)
        sample_range = sample_max - sample_min
        x_min = sample_min - sample_range / 10 if x_min is None else x_min
        x_max = sample_max + sample_range / 10 if x_max is None else x_max

    # watch out, scaling of weight_vec
    if weight_vec is not None:
        weight_vec = np.atleast_1d(weight_vec).squeeze()
        if np.any(weight_vec < 0):
            raise ValueError("Argument: weight_vec cannot have negative elements!")
        weight_vec = weight_vec / np.sum(weight_vec) * n_sample

    # Range of x_vec
    x_range = x_max - x_min

    # Histogram the sample_vec to get a crude first approximation of the density
    step = x_range / (n_x_vec - 1)
    x_vec = np.arange(start=x_min, stop=x_max + 0.1 * step, step=step)

    sample_hist, bin_edges = np.histogram(sample_vec, bins=x_vec, weights=weight_vec)
    # for easier comparison with Matlab, the count for [x_vec[-1], +Inf [ is also
    # added, i.e. 0
    sample_hist = np.append(sample_hist, 0)
    sample_hist = sample_hist / n_sample
    # discrete cosine transform of initial sample_vec
    dct_sample = fft.dct(sample_hist, norm=None)

    ic = np.arange(1, n_x_vec, dtype=float) ** 2
    sq_dct_sample = (dct_sample[1:] / 2) ** 2.0

    # The fixed point calculation finds the bandwidth = t_star
    guess = 0.1
    try:
        t_star = optimize.brentq(
            f=fixed_point, a=0, b=guess, args=(n_sample, ic, sq_dct_sample)
        )
    except (ValueError, RuntimeError) as e:
        logging.warning(
            "Failed to find the optimal bandwidth.\n\t"
            f"Root finding (Brent's method) failed with error: {e}.\n\t"
            "We fall back to use ``scipy.stats.gaussian_kde``).\n\t"
            "Please carefully check the results!"
        )
        # t_star = 0.28 * n_x_vec ** (-2 / 5)
        gkde = gaussian_kde(sample_vec, weights=weight_vec)
        density_vec = gkde.evaluate(x_vec)
        if return_bandwidth:
            return density_vec, x_vec, np.nan
        else:
            return density_vec, x_vec

    # Smooth the DCTransformed sample_vec using t_star
    sm_dct_sample = dct_sample * np.exp(
        -np.arange(n_x_vec) ** 2 * np.pi ** 2 * t_star / 2
    )
    # Inverse DCT to get density
    density_vec = fft.idct(sm_dct_sample, norm=None) / x_range
    bandwidth = np.sqrt(t_star) * x_range

    density_vec = density_vec / np.trapz(density_vec, x_vec)

    if return_bandwidth:
        return density_vec, x_vec, bandwidth
    else:
        return density_vec, x_vec


def fixed_point(t, n_sample, ic, sq_dct_sample):
    # this implements the function t-zeta*gamma**[l](t)
    c7 = 7
    ic = np.longdouble(ic)
    n_sample = np.longdouble(n_sample)
    sq_dct_sample = np.longdouble(sq_dct_sample)
    f = (
        2
        * np.pi ** (2 * c7)
        * np.sum(ic ** c7 * sq_dct_sample * np.exp(-ic * np.pi ** 2 * t))
    )
    for s in range(c7, 1, -1):
        k0 = np.prod(range(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = (2 * const * k0 / n_sample / f) ** (2 / (3 + 2 * s))
        f = (
            2
            * np.pi ** (2 * s)
            * np.sum(ic ** s * sq_dct_sample * np.exp(-ic * np.pi ** 2 * time))
        )
    return t - (2 * n_sample * np.sqrt(np.pi) * f) ** (-2 / 5)


# ======================================================================================
# 2D
# ======================================================================================
def kde_2d(
    sample_mx: Union[np.ndarray, list],
    n_row_mx: int = N_ROW_MX,
    xy_min: Union[np.ndarray, Iterable] = None,
    xy_max: Union[np.ndarray, Iterable] = None,
    weight_vec: Union[np.ndarray, list] = None,
    return_bandwidth: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, float],
]:
    """
    Fast and accurate state-of-the-art bivariate kernel density estimator with
    diagonal bandwidth matrix.

    The kernel is assumed to be Gaussian. The two bandwidth parameters are chosen
    optimally without ever using/assuming a parametric model for the sample_vec or
    any "rules of thumb". Unlike many other procedures, this one is immune to
    accuracy failures in the estimation of multimodal densities with widely separated
    modes.

    Args:
        sample_mx:
            A 2D matrix of sample_vec from which the density estimate is
            constructed, the matrix must have two columns that represent the two
            coordinates (x,y) of the 2D sample_vec.
        n_row_mx:
            Number of points along each dimension (same for columns) where the
            estimate of the density will be returned, i.e. total number of points is
            ``n_row_x_mx**2``.
        xy_min:
            The lower x and y boundaries of the interval over which the density
            estimate is constructed.
        xy_max:
            The upper x and y boundaries of the interval over which the density
            estimate is constructed.
        weight_vec:
            Weights of sample points. This must have the same number of
            elements as rows in ``sample_vec``, the same weight is applied to both
            coordinates of the same ``sample_vec`` point. If ``None`` (default),
            the samples are assumed to be equally weighted. The absolute value of the
            elements of ``weight_vec`` does not matter, only the values of elements
            relative to each other, i.e. multiplying ``weight_vec`` by a scalar does
            not change the results.
        return_bandwidth:
            Should the used bandwidth be returned?

    Raises:
        ValueError: If the number of columns in ``sample_mx`` is not two. If
            ``weight_vec`` has at least one negative value.

    Returns:
        Kernel densities, 2D matrix with the values of the density
        estimate at the grid points formed by ``x_mx`` and ``y_mx``.

        Kernel density grid (``x_mx``), the x coordinates of the grid points
        over which the density estimate is computed in the form of a 2D matrix
        that is the outcome of ``np.meshgrid``.

        Kernel density grid (``y_mx``), the x coordinates of the grid points
        over which the density estimate is computed in the form of a 2D matrix
        that is the outcome of ``np.meshgrid``.

        Optimal bandwidth (Gaussian kernel assumed), returned only if
        ``return_bandwidth`` is ``True``.

    .. note::
        To ease testing and debugging the implementation very closely follows [2],
        i.e. [2] is assumed to be correct.

    References:
        [1] Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010) Annals of
        Statistics, Volume 38, Number 5, pages 2916-2957.

        [2] https://nl.mathworks.com/matlabcentral/fileexchange/17204-kernel-density-estimation.  # noqa: E501
        Updated on: 30 Dec 2015.
    """

    sample_mx = np.atleast_2d(sample_mx)

    if sample_mx.shape[1] != 2:
        raise ValueError(
            f"``sample_vec`` should have exactly two columns but your input has:"
            f" {sample_mx.shape[1]}."
        )

    n_row_mx = int(2 ** np.ceil(np.log2(n_row_mx)))
    n_sample = sample_mx.shape[0]

    if xy_min is None or xy_max is None:
        xy_sample_max = np.max(sample_mx, axis=0)
        xy_sample_min = np.min(sample_mx, axis=0)
        xy_sample_range = xy_sample_max - xy_sample_min
        xy_max = xy_sample_max + xy_sample_range / 2 if xy_max is None else xy_max
        xy_min = xy_sample_min - xy_sample_range / 2 if xy_min is None else xy_min

    # watch out, scaling of weight_vec
    if weight_vec is not None:
        weight_vec = np.atleast_1d(weight_vec).squeeze()
        if np.any(weight_vec < 0):
            raise ValueError("Argument: ``weight_vec`` cannot have negative elements!")
        weight_vec = weight_vec / np.sum(weight_vec) * n_sample

    xy_max = np.atleast_1d(xy_max)
    xy_min = np.atleast_1d(xy_min)

    scaling = xy_max - xy_min
    transformed_sample = (sample_mx - xy_min) / scaling

    # bin the sample_vec uniformly using regular grid
    initial_sample = hist_2d(
        sample_mx=transformed_sample, n_bin=n_row_mx, weight_vec=weight_vec
    )

    # discrete cosine transform of initial sample_vec
    a = dct2d(initial_sample)

    # compute the optimal bandwidth**2
    ic = np.arange(start=0, stop=n_row_mx, step=1, dtype=float) ** 2
    ac2 = a ** 2
    t_star = root(
        lambda t: t - evolve(t, n_sample=n_sample, ic=ic, ac2=ac2)[0], n=n_sample
    )

    def func_(s, t):
        return func(s=s, t=t, n_sample=n_sample, ic=ic, ac2=ac2)

    p_02 = func_([0, 2], t_star)
    p_20 = func_([2, 0], t_star)
    p_11 = func_([1, 1], t_star)
    t_y = (
        p_02 ** (3 / 4)
        / (4 * np.pi * n_sample * p_20 ** (3 / 4) * (p_11 + np.sqrt(p_20 * p_02)))
    ) ** (1 / 3)
    t_x = (
        p_20 ** (3 / 4)
        / (4 * np.pi * n_sample * p_02 ** (3 / 4) * (p_11 + np.sqrt(p_20 * p_02)))
    ) ** (1 / 3)

    # smooth the discrete cosine transform of initial sample_vec using t_star
    n_range = np.arange(0, n_row_mx, dtype=float)
    v1 = np.atleast_2d(np.exp(-(n_range ** 2) * np.pi ** 2 * t_x / 2)).T
    v2 = np.atleast_2d(np.exp(-(n_range ** 2) * np.pi ** 2 * t_y / 2))
    a_t = np.matmul(v1, v2) * a

    # apply the inverse discrete cosine transform
    density_mx = idct2d(a_t) * (a_t.size / np.prod(scaling))
    # remove any negative density values
    density_mx[density_mx < 0] = np.finfo(float).eps
    x_step = scaling[0] / (n_row_mx - 1)
    y_step = scaling[1] / (n_row_mx - 1)
    x_vec = np.arange(start=xy_min[0], stop=xy_max[0] + 0.1 * x_step, step=x_step)
    y_vec = np.arange(start=xy_min[1], stop=xy_max[1] + 0.1 * y_step, step=y_step)
    x_mx, y_mx = np.meshgrid(x_vec, y_vec)
    bandwidth = np.sqrt([t_x, t_y]) * scaling

    density_mx = density_mx.T

    if return_bandwidth:
        return density_mx, x_mx, y_mx, bandwidth
    else:
        return density_mx, x_mx, y_mx


def evolve(t, n_sample: int, ic, ac2):
    def func_(ss, tt):
        return func(s=ss, t=tt, n_sample=n_sample, ic=ic, ac2=ac2)

    sum_func = func_([0, 2], t) + func_([2, 0], t) + 2 * func_([1, 1], t)
    time = (2 * np.pi * n_sample * sum_func) ** (-1 / 3)
    out = (t - time) / time
    return out, time


def func(s, t, n_sample, ic, ac2):
    if sum(s) <= 4:
        sum_func = func([s[0] + 1, s[1]], t, n_sample=n_sample, ic=ic, ac2=ac2) + func(
            [s[0], s[1] + 1], t, n_sample=n_sample, ic=ic, ac2=ac2
        )
        const = (1 + 1 / 2 ** (np.sum(s) + 1)) / 3
        time = (-2 * const * k_fun(s[0]) * k_fun(s[1]) / n_sample / sum_func) ** (
            1 / (2 + np.sum(s))
        )
        out = psi(s, time, ic, ac2)
    else:
        out = psi(s, t, ic, ac2)
    return out


def psi(s, time, ic, ac2):
    # s is a vector
    w = np.exp(-ic * np.pi ** 2 * time) * np.append(1, 0.5 * np.ones(len(ic) - 1))
    wx = w * (ic ** s[0])
    wy = w * (ic ** s[1])
    out = (
        (-1) ** np.sum(s)
        * (np.matmul(np.matmul(wy, ac2), wx.T))
        * np.pi ** (2 * np.sum(s))
    )
    return out


def k_fun(s):
    step = 2
    idx = np.arange(start=1, stop=2 * s - 1 + 0.1 * step, step=step)
    return (-1) ** s * np.prod(idx) / np.sqrt(2 * np.pi)


def dct2d(sample):
    # t_sample = fft.dct(fft.dct(sample_vec, axis=0), axis=1)
    t_sample = fft.dctn(sample)
    t_sample[:, 0] = t_sample[:, 0] / 2
    t_sample[0, :] = t_sample[0, :] / 2
    return t_sample


def idct2d(sample):
    sample = copy.deepcopy(sample)
    sample[:, 0] = sample[:, 0] * 2
    sample[0, :] = sample[0, :] * 2
    t_sample = fft.idctn(sample)
    return t_sample


def hist_2d(sample_mx, n_bin, weight_vec: Union[np.ndarray, list] = None) -> np.ndarray:
    """
    Computes the histogram of a 2-dimensional sample (two columns, n rows).

    Args:
        sample_mx: A sample of ``n_rows`` and ``n_columns``.
        n_bin: The number of bins used in each dimension so that ``binned_sample``
            is a hypercube with size length equal to ``n_bin``.
        weight_vec: Weights.

    Returns:
        Binned sample

    :meta private:
    """
    x = sample_mx[:, 0]
    y = sample_mx[:, 1]

    step = 1 / n_bin
    hc = np.histogram2d(
        x, y, bins=np.arange(0, 1 + 0.1 * step, step=step), weights=weight_vec
    )[0]
    binned_sample = hc / np.sum(hc)
    return binned_sample


def root(fun, n):
    # Try to find the smallest root whenever there is more than one.
    max_tol = 0.1
    n = 50 * int(n <= 50) + 1050 * int(n >= 1050) + n * int((n < 1050) & (n > 50))
    # pwith the current numbers this is at maximum 0.01
    tol = 10 ** -12 + 0.01 * (n - 50) / 1000

    solved = False
    while not solved:
        try:
            t = optimize.brentq(f=fun, a=0, b=tol)
            solved = True
        except ValueError:
            # double search interval
            tol = min(tol * 2, max_tol)

        # if all else fails
        if tol >= max_tol:
            t = optimize.fminbound(func=lambda x: abs(fun(x)), x1=0, x2=0.1)
            solved = True

    return t
