"""Main IDR functions."""
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats

from pyidr import _stats


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
