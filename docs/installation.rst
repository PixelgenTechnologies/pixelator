============
Installation
============


Stable release
--------------

To install pixelator, run this command in your terminal:

.. code-block:: console

    $ conda install -c bioconda pixelator

This is the preferred method to install pixelator, as it will always install the most recent stable release.

If you don't have `Anaconda`_ installed, this `Anaconda installation guide`_ can guide
you through the process.

.. _Anaconda: https://www.anaconda.com/
.. _Anaconda installation guide: https://www.anaconda.com/products/distribution


From source
-----------

The source for pixelator can be downloaded from the `GitHub repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/pixelgentechnologies/pixelator.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/pixelgentechnologies/pixelator/tarball/master

Once you have a copy of the source, follow the instructions in ../README.rst:

.. _GitHub repo: https://github.com/pixelgentechnologies/pixelator
.. _tarball: https://github.com/pixelgentechnologies/pixelator/tarball/master


Using the container image
--------------------------

A container image is available in the Pixelgen Github Container Registry.

.. code-block:: console

    $ docker pull ghcr.github.com/pixelgentechnologies/pixelator:latest

If you do not have a container engine installed you can follow the `Docker installation guide`_.

.. _`Docker installation guide`: https://docs.docker.com/engine/install/
