"""Base classes for IDR calculations."""
from typing import Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.distributions import empirical_distribution

from pyidr import _idr, _interpolations, _stats


class CorrespondenceProfile:
    # pylint: disable=too-many-instance-attributes
    """Compute the correspondence profile.

    Compute the correspondence profile for two ranked
    datasets with a given right-tail probability.

    """

    _jump_point: int
    _right_percent: np.ndarray
    _right_percent_binned: np.ndarray
    _uri: np.ndarray
    _spline: Dict[str, np.ndarray]
    _spline_slope: np.ndarray
    _spline_derivative: Dict[str, np.ndarray]
    _ntotal: int

    def __init__(self, ranking_1: np.ndarray, ranking_2: np.ndarray,
                 right_percent: np.ndarray):
        """Initialize the CorrespondenceProfile class.

        Args:
            ranking_1 (np.ndarray): First ranked dataset.
            ranking_2 (np.ndarray): Second ranked dataset.
            right_percent (np.ndarray): The right-tail percentage.
                A numeric vector between 0 and 1 in ascending order.
        """
        psi_dpsi = _interpolations.get_uri_2d(ranking_1, ranking_2,
                                              right_percent, right_percent)

        self._jump_point = psi_dpsi["jump_left"]
        self._right_percent = right_percent
        self._right_percent_binned = psi_dpsi["t_binned"]
        self._uri = psi_dpsi["uri"]
        self._spline = psi_dpsi["uri_spl"]
        self._spline_slope = psi_dpsi["uri_slope"]
        self._spline_derivative = psi_dpsi["uri_der"]
        self._ntotal = ranking_1.size
# yapf: disable
    def get_psi(
            self,
            scale: bool = False
        ) -> Dict[str, Union[Dict[str, np.ndarray], np.ndarray, float]]:
        # yapf: enable
        """Compute the correspondence profile.

            Correspondence profile as percentage or as number of
            observations when 'scale=True'.

        Args:
            scale (bool): scale by the total number of samples. Defaults to 'False'.

        Returns:
            Dict[str, Union[Dict[str, np.ndarray], np.ndarray, float]]:
            t:
                upper percentage for psi or number of
                top ranked observations
            value:
                psi
            smoothed_line:
                smoothing spline
            jump_point:
                the index of the vector of t such that psi(t[jump.point])
                jumps up due to ties at the low values. This only
                happends when data consists of a large number of discrete
                values, e.g. values imputed for observations
                appearing on only one replicate.
        """
        if scale:
            return dict(t=self._right_percent * self._ntotal,
                        value=self._uri * self._ntotal,
                        smoothed_line=dict(x=self._spline['x'] * self._ntotal,
                                           y=self._spline['y'] * self._ntotal),
                        jump_point=self._jump_point)
        return dict(t=self._right_percent,
                    value=self._uri,
                    smoothed_line=self._spline,
                    jump_point=self._jump_point)


# yapf: disable
    def get_dpsi(
            self,
            scale: bool = False
        ) -> Dict[str, Union[Dict[str, np.ndarray], np.ndarray, float]]:
        # yapf: enable
        """Compute the derivative of the correspondence profile.

            Derivative of correspondence profile as percentage or as number of
            observations when 'scale=True'.

        Args:
            scale (bool): scale by the total number of samples. Defaults to 'False'.

        Returns:
            Dict[str, Union[Dict[str, np.ndarray], np.ndarray, float]]:
            t:
                upper percentage for dpsi
            value:
                dpsi
            smoothed_line:
                smoothing spline
            jump_point:
                the index of the vector of t such that dpsi(t[jump.point])
                jumps up due to ties at the low values. This only
                happends when data consists of a large number of discrete
                values, e.g. values imputed for observations
                appearing on only one replicate.
        """
        if scale:
            return dict(t=self._right_percent_binned * self._ntotal,
                        value=self._spline_slope,
                        smoothed_line=dict(x=self._spline_derivative['x'] *
                                           self._ntotal,
                                           y=self._spline_derivative['y']),
                        jump_point=self._jump_point)
        return dict(t=self._right_percent_binned,
                    value=self._spline_slope,
                    smoothed_line=self._spline_derivative,
                    jump_point=self._jump_point)

    def plot_diagnostics(
            self) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plot Psi and Psi' vs  the right-tail percentage given as user input.

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
                The produced figure.
        """
        fig, axes = plt.subplots(2, 1, sharex=True)

        axes[0].scatter(self._right_percent[:self._jump_point],
                        self._uri[:self._jump_point])
        axes[0].plot(self._spline["x"], self._spline["y"])
        axes[0].plot((0, 1), (0, 1))
        axes[0].set_ylabel(r'$\Psi$')

        axes[1].plot(self._spline_derivative["x"],
                     self._spline_derivative["y"])
        axes[1].plot((0, 1), (1, 1))
        axes[1].set_ylabel(r"$\Psi'$")
        axes[1].set_xlabel(r't')
        return fig, axes


class IDR:
    # pylint: disable=too-many-instance-attributes
    """Compute the irreproducible recovery rate.

    Compute the irreproducible recovery rate (short idr) for two
    datasets.

    Attributes:
        mean (float): Mean value for the estimated gaussian.
        sigma (float): Sigma value for the estimated gaussian.
        rho (float): Correlation coefficient.
        prob (float): Mixing proportion of the reproducible component
        eps (float): Small epsilon used in calculations.
    """

    _mean: float
    _sigma: float
    _rho: float
    _prob: float
    _eps: float
    _maxiter: int
    _loglik_trace: List[float]

    def __init__(self,
                 mean: float = 2.6,
                 sigma: float = 1.3,
                 rho: float = 0.8,
                 prob: float = 0.7,
                 eps: float = 0.001,
                 maxiter: int = 30):
        """Initilize the IDR class with initial parameters.

        Args:
            mean (float): Initial value for mean. Defaults to 2.6.
            sigma (float): Initial value for standard deviation. Defaults to 1.3.
            rho (float): Initial value for correlation coefficient. Defaults to 0.8.
            prob (float): Initial value for mixing proportion of the reproducible component.
                Defaults to 0.7.
            eps (float): Small value used for computations. Defaults to 0.001.
            maxiter (int): Maximum number of fitting iterations. Defaults to 30.
        """
        # pylint: disable=too-many-arguments
        self._mean = mean
        self._sigma = sigma
        self._rho = rho
        self._prob = prob
        self._eps = eps
        self._maxiter = maxiter
        self._loglik_trace = list()

    @staticmethod
    def _setup(dataset1: np.ndarray,
               dataset2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dataset1 = dataset1.flatten()
        dataset2 = dataset2.flatten()
        assert dataset1.size == dataset2.size
        afactor = dataset1.size / (dataset1.size + 1)
        first_ecdf = empirical_distribution.ECDF(dataset1)(dataset1) * afactor
        second_ecdf = empirical_distribution.ECDF(dataset2)(dataset2) * afactor
        return first_ecdf, second_ecdf

    def fit(self, dataset1: np.ndarray, dataset2: np.ndarray) -> 'IDR':
        """Fit the parameter to the two datasets using copula mixture models.

        Args:
            dataset1 (np.ndarray): The first dataset.
            datasets (np.ndarray): The second dataset.

        Returns:
            IDR: The fitted IDR object (self).

        Raises:
            AssertionError: Error if both datasets have unequal length.
        """
        first_ecdf, second_ecdf = self._setup(dataset1, dataset2)
        parameter = dict(mean=self._mean,
                         sigma=self._sigma,
                         rho=self._rho,
                         prob=self._prob)
        parameter, self._loglik_trace, _ = _idr._fit_idr(  # pylint: disable=protected-access
            first_ecdf, second_ecdf, parameter, self._eps, self._maxiter)
        self._prob = parameter['prob']
        self._rho = parameter['rho']
        self._sigma = parameter['sigma']
        self._mean = parameter['mean']
        return self

    def predict(self, dataset1: np.ndarray,
                dataset2: np.ndarray) -> np.ndarray:
        """Predict the local idr for the two datasets.

        Args:
            dataset1 (np.ndarray): The first dataset.
            dataset2 (np.ndarray): The second dataset.

        Returns:
            np.ndarray: The local idr for each observation
            (i.e. estimated conditionalprobablility
            for each observation to belong to the irreproducible component.

        Raises:
            AssertionError: Error if both datasets have unequal length.
        """
        parameter = dict(mean=self._mean,
                         sigma=self._sigma,
                         rho=self._rho,
                         prob=self._prob)
        first_ecdf, second_ecdf = self._setup(dataset1, dataset2)
        z_1 = _stats.pseudo_mix(first_ecdf, **parameter)
        z_2 = _stats.pseudo_mix(second_ecdf, **parameter)
        e_z = _stats.e_two_normal(z_1, z_2, **parameter)
        return 1 - e_z

    def fit_predict(self, dataset1: np.ndarray,
                    dataset2: np.ndarray) -> np.ndarray:
        """Call fit and predict. Same as calling first 'fit' and then 'predict'.

        Args:
            dataset1 (np.ndarray): The first dataset.
            dataset2 (np.ndarray): The second dataset.

        Returns:
            np.ndarray: The  local idr for each observation
            (i.e. estimated conditionalprobablility for
            each observation to belong to the irreproducible component.

        Raises:
            AssertionError: Error if both datasets have unequal length.
        """
        self.fit(dataset1, dataset2)
        return self.predict(dataset1, dataset2)

    def predict_global(self, dataset1: np.ndarray,
                       dataset2: np.ndarray) -> np.ndarray:
        """Compute the expected idr for all observations.

        Args:
            dataset1 (np.ndarray): The first dataset.
            dataset2 (np.ndarray): The second dataset.

        Returns:
            np.ndarray: The expected irreproducible discovery rate for
            observations that are as irreproducible or more irreproducible
            than the given observations.

        Raises:
            AssertionError: Error if both datasets have unequal length.
        """
        return _idr._order_idr(self.predict(dataset1, dataset2))  # pylint: disable=protected-access

    def set_params(self, **params) -> 'IDR':
        """Set parameters to model.

        Allowed parameters are: 'prob','rho','mean','sigma','eps' and 'maxiter'.

        Returns:
            IDR: Updated IDR object (self).
        """
        updates = dict(_prob=params.get('prob', self._prob),
                       _rho=params.get('rho', self._rho),
                       _mean=params.get('mean', self._mean),
                       _sigma=params.get('sigma', self._sigma),
                       _eps=params.get('eps', self._eps),
                       _maxiter=params.get('maxiter', self._maxiter))
        self.__dict__.update(updates)
        return self

    def get_params(self) -> Dict[str, float]:
        """Get parameter from model.

        Returns:
            Dict[str, float]: All parameters.
        """
        return dict(prob=self._prob,
                    rho=self._rho,
                    mean=self._mean,
                    sigma=self._sigma,
                    eps=self._eps,
                    maxiter=self._maxiter)

    @property
    def proportion(self):
        """Return the proportion of reproducible component for a fitted model.

        Returns:
            float: The proportion of reproducible component.
        """
        return self._prob

    @staticmethod
    def get_correspondence(dataset1: np.ndarray, dataset2: np.ndarray,
                           right_percent: np.ndarray) -> CorrespondenceProfile:
        """Compute the correspondance profile.

        Ranks both datasets and computes the correspondence profile.

        Args:
            dataset1 (np.ndarray): The first dataset.
            dataset2 (np.ndarray): The second dataset.
            right_percent (np.ndarray): The right-tail percentage.
                A numeric vector between 0 and 1 in ascending order.

        Returns:
            CorrespondenceProfile: The correspondance profile.
        """
        ranking_1 = stats.rankdata(dataset1.flatten())
        ranking_2 = stats.rankdata(dataset2.flatten())
        return CorrespondenceProfile(ranking_1, ranking_2, right_percent)
