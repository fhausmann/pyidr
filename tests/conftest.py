"""Pytest config."""
import pytest
from rpy2.robjects.packages import importr  # pylint: disable=import-error


@pytest.fixture
def r_idr():
    """Import the R idr package."""
    idr = importr('idr')
    return idr
