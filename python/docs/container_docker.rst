.. _container_docker:

Docker, Podman & Kubernetes
===========================

Docker
------

For convenience, a Docker container can be used to run Python scripts
that make use of OSKAR without needing to compile or install any code.
The container is on Docker Hub and can be accessed under the tag
`docker.io/fdulwich/oskar-python3:latest` (or a named version).

The container should be deployed by attaching the directory containing
the input data files, settings and/or scripts as a bind mount to a
directory inside the container, for example to ``/data``.
(Output data files will also be written here.)
Using ``docker`` from the command line, the bind mount can be set using
the ``-v`` option, and the work directory inside the container specified
using the ``-w`` option.

To make sure the output files are created with the correct ownership
(since Docker runs everything as the root user by default), a user is
created on-the-fly by reading the value of the environment
variable ``LOCAL_USER_ID``. This should be set to the numeric ID of
the user running the script, and can be obtained by running ``id -u $USER``
in the shell. Run ``docker`` with the ``-e`` flag to pass the environment
variable into the container.

Combining all these to run a Python script called ``hello-world.py``
(from the :ref:`"hello world" example <example_hello_world>`) in the container,
use:

.. code-block:: bash

   docker run -v $PWD:/data -w /data -e LOCAL_USER_ID=`id -u $USER` \
       docker.io/fdulwich/oskar-python3 python3 hello-world.py

Podman
------

If using ``podman`` instead of ``docker`` to run the container, note that
filesystem permissions on bind mounts are handled differently between the two
systems. With ``docker``, files written as root inside the container will
appear as if they had been written by root on the host (not a good idea!),
whereas files written as root inside a container using ``podman``
will appear on the host as if they had been written by the user that started
the container.

To work around this issue when using ``docker``, an entrypoint script is used
to create a user on-the-fly with a matching numerical UID whenever the
container is started, and then executes code as that user. Although ``podman``
handles the mapping between users differently so this is no longer required,
unfortunately the same container started in the same way using ``podman`` will
break, as the newly-created user will not have permission to write to the
specified location on the host.
(Inside the container, this location will instead appear to be owned by the
root user.) Therefore if using ``podman``, the entrypoint script must be
bypassed by specifying an empty entrypoint when running the container,
as follows:

.. code-block:: bash

   podman run -v $PWD:/data -w /data --entrypoint= \
       docker.io/fdulwich/oskar-python3 python3 hello-world.py

Kubernetes
----------

To run a Python script using Kubernetes (for example on P3-ALaSKA), the shell
script below can be used to run the container as a Kubernetes Job by setting
the required parameters automatically. The helper script takes care of mounting
the current directory into the container, setting the work directory and
passing in the ``LOCAL_USER_ID`` environment variable. Use the command line
arguments to the script to specify what should be run inside the container.

For example, if the helper script is saved as ``oskar_run_k8s``, the example
Python script ``hello-world.py`` could be run using:

.. code-block:: bash

   oskar_run_k8s python3 hello-world.py

The helper script is included below (:download:`oskar_run_k8s`):

.. literalinclude:: oskar_run_k8s
   :language: bash
