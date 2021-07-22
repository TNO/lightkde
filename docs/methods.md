(sec:methods)=
# Methods

📔

## Overview

Diffusion based kernel density estimation after {cite}`botev2010`. This Python
implementation is based on Botev's MATLAB implementations for 1d and 2d
samples, {cite}`botev_kde` and {cite}`botev_kde2d` respectively. The 1d python
implementation is influenced by {cite}`smith_kde`.

A brief description of the implemented method:

> Reliable and extremely fast kernel density estimator for one-dimensional data;
Gaussian kernel is assumed and the bandwidth is chosen automatically;
Unlike many other implementations, this one is immune to problems
caused by multimodal densities with widely separated modes (see example). The
estimation does not deteriorate for multimodal densities, because we never assume
a parametric model for the data (like those used in rules of thumb).

{cite}`botev_kde`

## Differences to the MATLAB implementation

* Extended to be able to handle weighted samples.
* Extensive testing (mostly against the MATLAB implementation that is assumed to be
  correct).
* Additional examples ({ref}`examples`).
