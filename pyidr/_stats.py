"""Basic stats for IDR."""
from typing import Dict

import numpy as np
from scipy.stats import norm


def pseudo_mix(data: np.ndarray, mean: float, sigma: float, prob: float,
               **kwargs) -> np.ndarray:
    # pylint: disable=unused-argument
    """Compute the pseudo values of a mixture model from the empirical CDF.

    Args:
      data (np.ndarray): A vector of values of empirical CDF
      mean (float): Mean of the reproducible component in the mixture model on the
          latent space
      sigma (float): Standard deviation of the reproducible component in the mixture
          model on the latent space
      p (float): Mixing proportion of the reproducible component in the mixture model
         on the latent space

    Returns:
      np.ndarray: The values of a mixture model corresponding to the empirical CDF

    """
    total = 1000
    wval = np.linspace(start=min(-3, mean - 3 * sigma),
                       stop=max(mean + 3 * sigma, 3),
                       num=total)
    w_cdf = prob * norm.cdf(wval, loc=mean, scale=sigma) + (
        1 - prob) * norm.cdf(wval, loc=0, scale=1)

    quan_x = np.empty_like(data)

    for i in range(total - 1):
        index = np.logical_and(data >= w_cdf[i], data < w_cdf[i + 1])
        tmp = (data[index] - w_cdf[i]) * (wval[i + 1] - wval[i])
        tmp /= (w_cdf[i + 1] - w_cdf[i])
        tmp += wval[i]
        quan_x[index] = tmp

    quan_x[data < w_cdf[0]] = wval[0]
    quan_x[data > w_cdf[-1]] = wval[-1]
    return quan_x


_NEG_LOG_TWO_MINUS_PI: float = -np.log(2) - np.log(np.pi)


def _d_binormal(z_1: np.ndarray, z_2: np.ndarray, mean: float, sigma: float,
                rho: float) -> np.ndarray:
    """Compute the log-density for parameterized bivariate Gaussian distribution.

    Parameterized bivariate Gaussian distribution defined as:
        N(mean, mean, sigma, sigma, rho).

    Args:
        z_1 (np.ndarray):  a numerical data vector on coordinate 1.
        z_2 (np.ndarray): a numerical data vector on coordinate 2.
        mean (float): mean
        sigma (float):  standard deviation
        rho (float): correlation coefficient

    Returns:
        np.ndarray: Log density of bivariate Gaussian distribution
            N(mean, mean, sigma, sigma, rho).
    """
    rho_sq = rho**2
    diff_mean_1 = z_1 - mean
    diff_mean_2 = z_2 - mean
    first_part = _NEG_LOG_TWO_MINUS_PI - 2 * np.log(sigma)
    second_part = np.log(1 - rho_sq) / 2
    big_sum = diff_mean_1**2 - 2 * rho * diff_mean_1 * diff_mean_2 + diff_mean_2**2
    third_part = (0.5 / (1 - rho_sq) / sigma**2) * big_sum
    return first_part - second_part - third_part


def e_two_normal(z_1: np.ndarray, z_2: np.ndarray, prob: float, mean: float,
                 sigma: float, rho: float) -> np.ndarray:
    """Run the expectation step in the EM algorithm.

    Expectation step for parameterized bivariate
    2-component Gaussian mixture models with (1-p)N(0, 0, 1, 1, 0) +
    pN(mean, mean, sigma, sigma, rho).

    Args:
        z_1 (np.ndarray): a numerical data vector on coordinate 1.
        z_2 (np.ndarray): a numerical data vector on coordinate 2.
        prob (float): mixing proportion of the reproducible component.
        mean (float): mean for the reproducible component.
        sigma (float): standard deviation of the reproducible component.
        rho (float): correlation coefficient of the reproducible component.

    Returns:
        np.ndarray: a numeric vector, where each entry represents the estimated expected
            conditional probability that an observation is in the reproducible
            component.
    """
    # pylint: disable=too-many-arguments
    first_binormal = _d_binormal(z_1, z_2, 0, 1, 0)
    second_binormal = _d_binormal(z_1, z_2, mean, sigma, rho)
    exp = np.exp(first_binormal - second_binormal)
    return prob / ((1 - prob) * exp + prob)


def m_two_normal(z_1: np.ndarray, z_2: np.ndarray,
                 e_z: np.ndarray) -> Dict[str, float]:
    """Run the maximization step in the EM algorithm.

    Maximization step for parameterized bivariate
        2-component Gaussian mixture models with $(1-p)N(0, 0, 1, 1, 0) +
        pN(mean, mean, sigma**2, sigma**2, rho)$.

    Args:
        z_1 (np.ndarray): a numerical data vector on coordinate 1.
        z_2 (np.ndarray): a numerical data vector on coordinate 2.
        e_z (np.ndarray): a vector of expected conditional probability that the $i$th
            observation is reproducible.

    Returns:
        Dict[str, float]: Estimated parameters, basically:
            p: the estimated mixing proportion of the reproducible component.
             mean: the estimated mean for the reproducible component.
            sigma: the estimated standard deviation of the reproducible component.
            rho: the estimated correlation coefficient of the reproducible component.
    """
    estimate_p = np.mean(e_z)
    estimate_mean = ((z_1 + z_2) * e_z).sum() / 2 / e_z.sum()
    diff_mean_1 = z_1 - estimate_mean
    diff_mean_2 = z_2 - estimate_mean
    diff_mean_1_sq = diff_mean_1**2
    diff_mean_2_sq = diff_mean_2**2
    sigma_first_sum = np.sum(e_z * (diff_mean_1_sq + diff_mean_2_sq))
    estimate_sigma = np.sqrt(sigma_first_sum / 2 / np.sum(e_z))
    estimate_rho = 2 * np.sum(e_z * diff_mean_1 * diff_mean_2) / (np.sum(
        e_z * (diff_mean_1_sq + diff_mean_2_sq)))
    return dict(prob=estimate_p,
                mean=estimate_mean,
                sigma=estimate_sigma,
                rho=estimate_rho)


def two_binormal(z_1: np.ndarray, z_2: np.ndarray, mean: float, sigma: float,
                 rho: float, prob: float) -> float:
    """Compute the log-likelihood for models.

    As models parameterized bivariate 2-component Gaussian
    mixture models with (1-p)N(0, 0, 1, 1, 0) + pN(mean, mean, sigma,sigma, rho)
    are used.

    Args:
        z_1 (np.ndarray): a numerical data vector on coordinate 1.
        z_2 (np.ndarray): a numerical data vector on coordinate 2.
        mean (float): mean for the reproducible component.
        sigma (float): standard deviation of the reproducible component.
        rho (float): correlation coefficient of the reproducible component.
        prob (float): mixing proportion of the reproducible component.

    Returns:
        float: Log-likelihood of the bivariate 2-component Gaussian mixture models
                $(1-p)N(0, 0, 1, 1, 0) + N(mean, mean, sigma, sigma, rho)$.
    """
    # pylint: disable=too-many-arguments
    binorm = _d_binormal(z_1, z_2, 0, 1, 0)
    exp = np.exp(_d_binormal(z_1, z_2, mean, sigma, rho) - binorm)
    loglik = binorm + np.log(prob * exp + (1 - prob))
    return np.sum(loglik)
