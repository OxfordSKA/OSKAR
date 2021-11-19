.. raw:: latex

    \clearpage

.. _containers:

Using Containers
================

Container images for `Docker <https://www.docker.com/>`_ and
`Singularity <https://sylabs.io/singularity/>`_ have been built to allow OSKAR
to be used easily on systems where these are deployed.

To run an OSKAR Python script on P3-ALaSKA using Kubernetes, see the example
given in the :ref:`container_docker` section.

To run an OSKAR Python script on a HPC system where Singularity is available,
see the example given in the :ref:`container_singularity` section.

.. note::

  These containers are intended for use only on Linux systems (optionally)
  with NVIDIA GPUs, as they use the `nvidia/cuda` base image.
  However, they will also run on Linux systems without GPUs attached.

  On the host system, GPU drivers supporting CUDA 11.4 or later must be
  installed in order to use these containers with GPUs.


.. toctree::
   :maxdepth: 2

   container_docker
   container_singularity
