==================================================================
pyidr - Python implementation of the irreproducible discovery rate
==================================================================
.. image:: https://github.com/fhausmann/pyidr/workflows/test/badge.svg?branch=master
    :target: https://github.com/fhausmann/pyidr/workflows/test/badge.svg?branch=master
    :alt: Pipeline status

.. image:: https://readthedocs.org/projects/pyidr/badge/?version=stable
    :target: https://pyidr.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation Status

Python implementation of of the irreproducible discovery rate (short IDR)
proposed in [LI2011]_.

IDR was originally published as R package idr__.

.. __: https://cran.r-project.org/web/packages/idr/index.html

Basic usage
===========

The local IDR and the proportion of the reproducibility component
can be done as follows::

    import pyidr
    idr = pyidr.IDR()
    idr.fit(dataset1, dataset2)
    local_idr = idr.predict(dataset1,dataset2)
    proportion_of_reproducibility = idr.proportion

The correspondence profile used for diagnostics can be caluclated and plotted
with the :class:`CorrespondenceProfile` class as follows::

    import pyidr
    profile = pyidr.CorrespondenceProfile(rankx, ranky, t)
    profile.plot_diagnostics()


.. [LI2011] Li, Qunhua, et al.
    "Measuring reproducibility of high-throughput experiments."
    The annals of applied statistics 5.3 (2011): 1752-1779.
