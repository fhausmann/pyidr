"""Tests pyidr base classes."""
import numpy as np
import pytest
import rpy2.robjects as ro

from pyidr import _base as base

#pylint: disable=unused-argument


@pytest.mark.parametrize("dataset", ["simu.idr"])
def test_idr(r_idr, dataset):
    """Test the idr function."""
    ro.r('data("{}")'.format(dataset))
    ro.r('x <- cbind(-{dataset}$x, -{dataset}$y)'.format(dataset=dataset))
    ro.globalenv['mean'] = 2.6
    ro.globalenv['sigma'] = 1.3
    ro.globalenv['rho'] = 0.8
    ro.globalenv['prob'] = 0.7
    expected_func = ro.r["est.IDR"]
    expected = expected_func(ro.r['x'], ro.r['mean'], ro.r['sigma'],
                             ro.r['rho'], ro.r['prob'])
    inputs = np.array(ro.r['x'])
    idrclass = base.IDR(mean=ro.r['mean'][0],
                        sigma=ro.r['sigma'][0],
                        rho=ro.r['rho'][0],
                        prob=ro.r['prob'][0])

    got = idrclass.fit_predict(inputs[:, 0], inputs[:, 1])
    expected_param = expected[0]
    params = idrclass.get_params()
    np.testing.assert_allclose(params['prob'], expected_param[0])
    np.testing.assert_allclose(params['rho'], expected_param[1])
    np.testing.assert_allclose(params['mean'], expected_param[2])
    np.testing.assert_allclose(params['sigma'], expected_param[3])
    np.testing.assert_allclose(idrclass._loglik_trace[-1], expected[1])  # pylint: disable=protected-access
    np.testing.assert_allclose(idrclass._loglik_trace, expected[2], rtol=1e-3)  # pylint: disable=protected-access
    np.testing.assert_allclose(got, expected[3], rtol=0.3)
    expected_idr = idrclass.predict_global(inputs[:, 0], inputs[:, 1])
    np.testing.assert_allclose(expected_idr, expected[4], rtol=0.3)
    idrclass.get_correspondence(inputs[:, 0], inputs[:, 1],
                                np.arange(0.01, 0.99, step=1 / 28))
    new_params = dict(prob=0, mean=1, sigma=5, eps=0.001, maxiter=10, rho=0.6)
    idrclass.set_params(**new_params)
    for key, value in idrclass.get_params().items():
        assert new_params[key] == value


@pytest.mark.parametrize("dataset", ["simu.idr"])
def test_get_correspondence(r_idr, dataset):
    """Test the get_correspondance function."""
    ro.r('data("{}")'.format(dataset))
    ro.r('x <- rank(-{dataset}$x)'.format(dataset=dataset))
    ro.r('y <- rank(-{dataset}$y)'.format(dataset=dataset))
    ro.r('t <- seq(0.01,0.99,by=1/28)')
    expected_func = ro.r["get.correspondence"]
    expected = expected_func(ro.r['x'], ro.r['y'], ro.r['t'])
    inputs = (np.array(ro.r['x']), np.array(ro.r['y']), np.array(ro.r['t']))
    idrclass = base.IDR(mean=0, sigma=0, rho=0, prob=0)
    correspondence = idrclass.get_correspondence(*inputs)
    psi = correspondence.get_psi()
    np.testing.assert_allclose(psi['t'], expected[0][0])
    np.testing.assert_allclose(psi['value'], expected[0][1], atol=0.002)
    np.testing.assert_allclose(psi['smoothed_line']['x'],
                               expected[0][2][0][:-1])
    np.testing.assert_allclose(psi['smoothed_line']['y'],
                               expected[0][2][1][:-1],
                               atol=0.01)
    np.testing.assert_allclose(psi['jump_point'], expected[0][4], atol=1)
    dpsi = correspondence.get_dpsi()
    np.testing.assert_allclose(dpsi['t'], expected[1][0])
    np.testing.assert_allclose(dpsi['value'], expected[1][1], atol=0.007)
    np.testing.assert_allclose(dpsi['smoothed_line']['x'],
                               expected[1][2][0][:-1])
    np.testing.assert_allclose(dpsi['smoothed_line']['y'],
                               expected[1][2][1][:-1],
                               atol=0.36)
    psi = correspondence.get_psi(scale=True)
    np.testing.assert_allclose(psi['t'], expected[2][0])
    np.testing.assert_allclose(psi['value'], expected[2][1], atol=1.)
    np.testing.assert_allclose(psi['smoothed_line']['x'],
                               expected[2][2][0][:-1])
    np.testing.assert_allclose(psi['smoothed_line']['y'][1:],
                               expected[2][2][1][1:-1],
                               rtol=0.06)
    dpsi = correspondence.get_dpsi(scale=True)
    np.testing.assert_allclose(dpsi['t'], expected[3][0])
    np.testing.assert_allclose(dpsi['value'], expected[3][1], atol=0.007)
    np.testing.assert_allclose(dpsi['smoothed_line']['x'],
                               expected[3][2][0][:-1])
    np.testing.assert_allclose(dpsi['smoothed_line']['y'],
                               expected[3][2][1][:-1],
                               atol=0.36)
