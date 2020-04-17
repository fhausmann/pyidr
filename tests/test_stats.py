"""Tests for stats submodule."""
import numpy as np
import rpy2.robjects as robjects

from pyidr import _stats as stats

# pylint: disable=unused-argument


def test_pseudo_mix(r_idr):
    """Test pseudo_mix function."""
    expected_func = robjects.r["get.pseudo.mix"]
    robjects.globalenv['rho'] = 0.
    for _ in range(10):
        inputs = np.random.rand(1000)
        mean, sigma, pval = np.random.rand(3)
        robjects.globalenv['mean'] = float(mean)
        robjects.globalenv['sigma'] = float(sigma)
        robjects.globalenv['prob'] = float(pval)
        inputs_r = robjects.FloatVector(inputs)
        got = stats.pseudo_mix(inputs, mean, sigma, pval)
        np.testing.assert_equal(inputs.shape, got.shape)
        expected = expected_func(inputs_r, robjects.r['mean'],
                                 robjects.r['sigma'], robjects.r['rho'],
                                 robjects.r['prob'])
        np.testing.assert_allclose(got, expected)


def test__d_binormal(r_idr):
    """Test _d_binormal function."""
    expected_func = robjects.r["d.binormal"]
    for _ in range(10):
        inputs1 = np.random.rand(1000)
        inputs2 = np.random.rand(1000)
        mean, sigma, rho = np.random.rand(3)
        robjects.globalenv['mean'] = float(mean)
        robjects.globalenv['sigma'] = float(sigma)
        robjects.globalenv['rho'] = float(rho)
        inputs_r1 = robjects.FloatVector(inputs1)
        inputs_r2 = robjects.FloatVector(inputs2)
        got = stats._d_binormal(inputs1, inputs2, mean, sigma, rho)  # pylint: disable=protected-access
        expected = expected_func(
            inputs_r1,
            inputs_r2,
            robjects.r['mean'],
            robjects.r['sigma'],
            robjects.r['rho'],
        )
        expected = np.array(expected)
        np.testing.assert_equal(got.shape, expected.shape)
        np.testing.assert_allclose(got, expected)


def test_e_two_normal(r_idr):
    """Test e_two_normal function."""
    expected_func = robjects.r["e.step.2normal"]
    for _ in range(10):
        inputs1 = np.random.rand(100)
        inputs2 = np.random.rand(100)
        mean, sigma, rho, pval = np.random.rand(4)
        robjects.globalenv['mean'] = float(mean)
        robjects.globalenv['sigma'] = float(sigma)
        robjects.globalenv['rho'] = float(rho)
        robjects.globalenv['prob'] = float(pval)
        inputs_r1 = robjects.FloatVector(inputs1)
        inputs_r2 = robjects.FloatVector(inputs2)
        got = stats.e_two_normal(inputs1, inputs2, pval, mean, sigma, rho)
        expected = expected_func(
            inputs_r1,
            inputs_r2,
            robjects.r['mean'],
            robjects.r['sigma'],
            robjects.r['rho'],
            robjects.r['prob'],
        )
        expected = np.array(expected)
        np.testing.assert_equal(got.shape, expected.shape)
        np.testing.assert_allclose(got, expected)


def test_two_binormal(r_idr):
    """Test two_normal function."""
    expected_func = robjects.r["loglik.2binormal"]
    for _ in range(10):
        inputs1 = np.random.rand(1000)
        inputs2 = np.random.rand(1000)
        mean, sigma, rho, pval = np.random.rand(4)
        robjects.globalenv['mean'] = float(mean)
        robjects.globalenv['sigma'] = float(sigma)
        robjects.globalenv['rho'] = float(rho)
        robjects.globalenv['prob'] = float(pval)
        inputs_r1 = robjects.FloatVector(inputs1)
        inputs_r2 = robjects.FloatVector(inputs2)
        got = stats.two_binormal(inputs1, inputs2, mean, sigma, rho, pval)
        expected = expected_func(
            inputs_r1,
            inputs_r2,
            robjects.r['mean'],
            robjects.r['sigma'],
            robjects.r['rho'],
            robjects.r['prob'],
        )
        np.testing.assert_allclose(got, expected[0])


def test_m_two_normal(r_idr):
    """Test m_two_normal function."""
    expected_func = robjects.r["m.step.2normal"]
    for _ in range(10):
        inputs1 = np.random.rand(1000)
        inputs2 = np.random.rand(1000)
        inputs3 = np.random.rand(1000)
        inputs_r1 = robjects.FloatVector(inputs1)
        inputs_r2 = robjects.FloatVector(inputs2)
        inputs_r3 = robjects.FloatVector(inputs3)
        got = stats.m_two_normal(inputs1, inputs2, inputs3)
        exp_p, exp_mean, exp_sigma, exp_rho = expected_func(
            inputs_r1, inputs_r2, inputs_r3)
        np.testing.assert_allclose(got["prob"], exp_p[0])
        np.testing.assert_allclose(got["mean"], exp_mean[0])
        np.testing.assert_allclose(got["sigma"], exp_sigma[0])
        np.testing.assert_allclose(got["rho"], exp_rho[0])
