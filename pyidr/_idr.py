"""Main IDR functions."""
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats
from statsmodels.distributions import empirical_distribution

from pyidr import _interpolations, _stats


def _conv(old, new, eps):
    return abs(new - old) < eps * (1 + abs(new))


def _order_idr(idr_val: np.ndarray) -> np.ndarray:
    order = np.argsort(idr_val)
    ordered_idrs = idr_val[order]
    ranks = sp_stats.rankdata(ordered_idrs, method="max") - 1
    mean_idrs = np.empty_like(idr_val)
    for i, idx in enumerate(ranks):
        mean_idrs[i] = ordered_idrs[:idx + 1].mean()
    ordered_idr = idr_val
    ordered_idr[order] = mean_idrs
    return ordered_idr


def _fit_idr(first_ecdf: np.ndarray, second_ecdf: np.ndarray,
             parameter: Dict[str, float], eps: float,
             maxiter: int) -> Tuple[Dict[str, float], List[float], np.ndarray]:
    loglik_trace: List[float] = list()
    z_1 = _stats.pseudo_mix(first_ecdf, **parameter)
    z_2 = _stats.pseudo_mix(second_ecdf, **parameter)
    l_new_outer = np.inf
    l_old_outer = np.nan
    j = 0
    while not _conv(l_old_outer, l_new_outer, eps) and j < maxiter:
        loglik_inner: float = np.nan
        l_new = np.inf
        while not _conv(loglik_inner, l_new, eps):
            loglik_inner = l_new
            e_z = _stats.e_two_normal(z_1, z_2, **parameter)
            parameter = _stats.m_two_normal(z_1, z_2, e_z)
            l_new = _stats.two_binormal(z_1, z_2, **parameter)
        z_1 = _stats.pseudo_mix(first_ecdf, **parameter)
        z_2 = _stats.pseudo_mix(second_ecdf, **parameter)
        l_old_outer = l_new_outer
        l_new_outer = _stats.two_binormal(z_1, z_2, **parameter)
        loglik_trace.append(l_new)
        j += 1
    return parameter, loglik_trace, e_z


# yapf: disable
def idr(
        data: np.ndarray,
        mean: float,
        sigma: float,
        rho: float,
        prob: float,
        eps: float = 0.001,
        maxiter: int = 30
    ) -> Tuple[Dict[str, float], float, List[float], np.ndarray, np.ndarray]:
    # yapf: enable
    """Estimate parameters from the copula mixture models.

    Args:
        data (np.ndarray): a m x 2 numeric matrix, where m is the number of replicates.
        mean (float): starting value for mean.
        sigma (float): starting value for standard deviation.
        rho (float): starting value for correlation coefficient.
        prob (float): starting value for mixing proportion of the reproducible component.

    Returns:
        Dict[str, float],float,List[float],np.ndarray,np.ndarray:
        estimated parameters:
            prob, rho, mean, sigma
        log-likelihood:
            log-likelihood at the end of iterations
        Trajectory:
            trajectory of log-likelihood
        Local idr for each observation:
            Estimated conditional probablility for each observation
            to belong to the irreproducible component.
        Expected irreproducible discovery rate:
            Expected irreproducible discovery rate for observations that
            are as irreproducible or more irreproducible
            than the given observations.

    """
    # pylint: disable=too-many-arguments, too-many-locals
    warnings.warn("Direct use of idr is deprecated. Use the IDR class instead",
                  category=DeprecationWarning,
                  stacklevel=2)
    afactor = data.shape[0] / (data.shape[0] + 1)
    first_ecdf = empirical_distribution.ECDF(data[:, 0])(data[:, 0]) * afactor
    second_ecdf = empirical_distribution.ECDF(data[:, 1])(data[:, 1]) * afactor
    parameter = dict(mean=mean, sigma=sigma, rho=rho, prob=prob)
    parameter, loglik_trace, e_z = _fit_idr(first_ecdf, second_ecdf, parameter,
                                            eps, maxiter)
    return parameter, loglik_trace[-1], loglik_trace, 1 - e_z, _order_idr(1 -
                                                                          e_z)


def get_correspondence(ranking_1: np.ndarray, ranking_2: np.ndarray,
                       right_percent: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Compute the correspondence profile.

    Args:
        ranking_1 (np.ndarray): Data values or ranks of the data values on list 1, a vector of
            numeric values. Large values need to be significant signals. If small
            values represent significant signals, rank the signals reversely
            (e.g. by ranking negative values) and use the rank as ranking_1.
        ranking_2 (np.ndarray): Data values or ranks of the data values on list 2, a vector of
            numeric values. Large values need to be significant signals. If small
            values represent significant signals, rank the signals reversely
            (e.g. by ranking negative values) and use the rank as ranking_2.
        right_percent (np.ndarray): A numeric vector between 0 and 1 in ascending order. t is the
            right-tail percentage.

    Returns:
        Dict[str, Dict[str, Any]]:
        psi:
            the correspondence profile in terms of the scale of percentage,
            i.e. between (0, 1)

        dpsi:
            the derivative of the correspondence profile in terms of the scale
            of percentage, i.e. between (0, 1)}

        psi_n:
            the correspondence profile in terms of the scale of the number of
            observations

        dpsi_n:
            the derivative of the correspondence profile in terms of the scale
            of the number of observations}

        Each object above is a dictonary consisting of the following items:

            t:
                upper percentage (for psi and dpsi) or number of
                top ranked observations (for psi_n and dpsi_n)

            value: psi or dpsi

            smoothed_line: smoothing spline

            ntotal: the number of observations

            jump_point:
                the index of the vector of t such that psi(t[jump.point])
                jumps up due to ties at the low values. This only
                happends when data consists of a large number of discrete
                values, e.g. values imputed for observations
                appearing on only one replicate.

    """
    warnings.warn(
        "Direct use of get_correspondence is deprecated."
        "Use the CorrespondenceProfile class instead",
        category=DeprecationWarning,
        stacklevel=2)
    psi_dpsi = _interpolations.get_uri_2d(ranking_1, ranking_2, right_percent,
                                          right_percent)
    ntotal = ranking_1.size
    psi = dict(t=right_percent,
               value=psi_dpsi["uri"],
               smoothed_line=psi_dpsi["uri_spl"],
               jump_point=psi_dpsi["jump_left"])
    dpsi = dict(t=psi_dpsi["t_binned"],
                value=psi_dpsi["uri_slope"],
                smoothed_line=psi_dpsi["uri_der"],
                jump_point=psi_dpsi["jump_left"])

    psi_n = dict(t=right_percent * ntotal,
                 value=psi["value"] * ntotal,
                 smoothed_line=dict(x=psi["smoothed_line"]["x"] * ntotal,
                                    y=psi["smoothed_line"]["y"] * ntotal),
                 jump_point=psi["jump_point"])

    dpsi_n = dict(t=dpsi["t"] * ntotal,
                  value=dpsi["value"],
                  smoothed_line=dict(x=dpsi["smoothed_line"]["x"] * ntotal,
                                     y=dpsi["smoothed_line"]["y"]),
                  jump_point=dpsi["jump_point"])
    return dict(psi=psi, dpsi=dpsi, psi_n=psi_n, dpsi_n=dpsi_n)
