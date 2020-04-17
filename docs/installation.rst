============
Installation
============

pyidr is available as Python package on PyPi and can be installed using pip::

    pip install pyidr

For advanced users and developers you can also install any version of pyidr
from the pyidr__ repository using poetry__.

.. __: https://github.com/fhausmann/pyidr
.. __: https://python-poetry.org/

For installation of pyidr from the repository use::

    git clone https://github.com/fhausmann/pyidr
    cd pyidr
    poetry install --no-dev

for the regular version or for the developmental version::

    git clone -branch devel https://github.com/fhausmann/pyidr
    cd pyidr
    poetry install

pyidr was tested with Python 3.7 and Python3.8.

Running the unit tests requires the developmental version of pyidr and
an R environment with the idr__ R package installed.

.. __: https://cran.r-project.org/web/packages/idr/index.html
