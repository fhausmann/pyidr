=====
Usage
=====
The usage heavily depends on our application but these are the most
probable usecases.

.. seealso:: For further insights what is possible, please have a look into to
    the :ref:`Function and class documentation`.

Calculation of local IDR
========================

The calculation if the local idr can be done with an sklearn__ like interface.

 .. __: https://scikit-learn.org/stable/index.html

To calculate the IDR the :class:`IDR` needs to be fitted to the data.

    idr_obj = pyidr.IDR(mean=0.1, var=0.4, maxiter=100)
    idr_obj.fit(dataset1, dataset2)

.. warning:: Significant signals are expected to have large values in
    both datasets. In case that smaller values represent higher significance
    (e.g.  p-value), a monotonic transformation needs to be applied to
    reverse the order before using this function, for example,`-log(p-value)`.

During the initialization of the class the default parameters can be changed.
Afterwards you can get the local IDR by using::

    local_idr = idr_obj.predict(dataset1,dataset2)

or in short::

    idr_obj = pyidr.IDR(mean=0.1, var=0.4, maxiter=100)
    local_idr = idr_obj.fit_predict(dataset1,dataset2)

Now you can get the proportion of the reproducibility component or decay point::

    proportion = idr_obj.proportion


Diagnostic correspondence profiles
==================================

 For diagnostic purposes you can visualize plots as in [LI2011]_ using the
 :class:`CorrespondenceProfile` class::

    import pyidr
    profile = pyidr.CorrespondenceProfile(rankx, ranky, t)
    profile.plot_diagnostics()

.. warning:: Large values need to be significant signals.
    If small values represent significant signals, rank the signals reversely
    (e.g. by ranking negative values) and use this as `rankx` and `ranky`.

For for deeper information you can get several diagnostic values by using::

    profile.get_psi(scale=scale)
    profile.get_dpsi(scale=scale)

`scale` is here a boolean indicating if the resulting values should be scaled
by the number of samples used for calculation.
