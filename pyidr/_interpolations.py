"""Module for spline interpolation."""
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import interpolate


def get_uri_2d(ranking_1: np.ndarray, ranking_2: np.ndarray,
               right_percent_1: np.ndarray,
               right_percent_2: np.ndarray) -> Dict[str, Any]:
    """Compute splines and the value of Psi and Psi'.

    Args:
        ranking_1 (np.ndarray): Data values or ranks of the data values on list 1, a vector of
            numeric values. Large values need to be significant signals. If small
            values represent significant signals, rank the signals reversely
            (e.g. by ranking negative values) and use the rank as ranking_1.
        ranking_2 (np.ndarray): Data values or ranks of the data values on list 2, a vector of
            numeric values. Large values need to be significant signals. If small
            values represent significant signals, rank the signals reversely
            (e.g. by ranking negative values) and use the rank as ranking_2.
        right_percent_1 (np.ndarray): A numeric vector between 0 and 1 in ascending order. t is the
            right-tail percentage for ranking_1.
        right_percent_2 (np.ndarray):  A numeric vector between 0 and 1 in ascending order. t is the
            right-tail percentage for ranking_2.

    Returns:
        Dict[str, Any]: Spline and the value of Psi and Psi'.
    """
    order_df = pd.DataFrame([ranking_1, ranking_2],
                            index=["ranking_1", "ranking_2"]).T.sort_values(
                                by=["ranking_1", "ranking_2"],
                                axis=0,
                                ascending=False)
    right_percent = np.vstack([right_percent_1, right_percent_2])
    uri = np.apply_along_axis(comp_uri,
                              axis=0,
                              arr=right_percent,
                              data=order_df["ranking_2"].values)
    # compute the derivative of URI vs t using small bins
    uri_binned = uri[::4]
    tt_binned = right_percent_1[::4]
    uri_slope = (uri_binned[1:] - uri_binned[:-1]) / (tt_binned[1:] -
                                                      tt_binned[:-1])
    # smooth uri using spline
    # first find where the jump is and don't fit the jump
    # this is the index on the left
    length = min(
        sum(ranking_1 > 0) / ranking_1.size,
        sum(ranking_2 > 0) / ranking_2.size)

    if length < right_percent_1.max():
        jump_left = np.flatnonzero(right_percent_1 > length)[0] - 1
    else:
        jump_left = right_percent_1.argmax()
    if jump_left < 5:
        jump_left = right_percent_1.size
    uri_spl = interpolate.UnivariateSpline(right_percent_1[:jump_left],
                                           uri[:jump_left],
                                           k=5)
    uri_der = uri_spl(right_percent_1[:jump_left], 1)
    uri_spl_y = uri_spl(right_percent_1[:jump_left])
    return dict(uri=uri,
                uri_slope=uri_slope,
                t_binned=tt_binned[1:],
                uri_spl=dict(x=right_percent_1[:jump_left], y=uri_spl_y),
                uri_der=dict(x=right_percent_1[:jump_left], y=uri_der),
                jump_left=jump_left)


def comp_uri(right_percent: np.ndarray, data: np.ndarray) -> float:
    """Compute PSI value.

    Args:
        right_percent (np.ndarray): A vector of two numeric values, t and v.
            Both t and v are in [0,1]
        data (np.ndarray): A numeric vector of x, sorted by the order of y
    Returns:
        [float]: A numeric value of Psi(t, v)
    """
    size = data.shape[0]
    quantile = np.quantile(data,
                           q=1 - right_percent[0])  # right_percent[0] is t
    idx = int(size * right_percent[1])
    return np.sum(data[:idx] >= quantile) / size
