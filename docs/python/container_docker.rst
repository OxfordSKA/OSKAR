.. _container_docker:

Docker & Kubernetes
===================

Docker
------

For convenience, a Docker container can be used to run Python scripts
that make use of OSKAR without needing to compile or install any code.
The container can be accessed using the tag `artefact.skao.int/oskar-python3`.

The container should be deployed by attaching the directory containing
the input data files, settings and/or scripts as a bind mount to a
directory inside the container, for example to ``/data``.
(Output data files will also be written here.)
Using ``docker`` from the command line, the bind mount can be set using
the ``-v`` option, and the work directory inside the container specified
using the ``-w`` option. The ``--user`` flag should be used to run the script
as the current user.

Combining all these to run a Python script called ``hello-world.py``
(from the :ref:`"hello world" example <example_hello_world>`) in the container,
use:

.. code-block:: bash

   docker run -v $PWD:/data -w /data --user `id -u`:`id -g` \
       artefact.skao.int/oskar-python3 python3 hello-world.py

Kubernetes
----------

To run a Python script using Kubernetes (for example on P3-ALaSKA), the shell
script below can be used to run the container as a Kubernetes Job by setting
the required parameters automatically. The helper script takes care of mounting
the current directory into the container and setting the work directory.
Use the command line arguments to the script (after a ``--``) to specify what
should be run inside the container.

For example, if the helper script is saved as ``oskar_run_k8s``, the example
Python script ``hello-world.py`` could be run using:

.. code-block:: bash

   oskar_run_k8s -- python3 hello-world.py

The helper script is included below (:download:`oskar_run_k8s`):

.. literalinclude:: oskar_run_k8s
   :language: bash
