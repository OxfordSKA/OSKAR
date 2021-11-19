.. _container_singularity:

Singularity
===========

`Singularity <https://sylabs.io/singularity/>`_ containers are more suitable
for use in traditional multi-user HPC environments, and are generally easier
to use than Docker containers for running straightforward applications.
Processes in a Singularity container run as the current user so there are fewer
security concerns, and the current directory is mounted by default, so binding
volumes is unnecessary.

For convenience, a SIF-format Singularity container (for use with
Singularity 3.0 and above) can be used to run Python scripts that make use of
OSKAR without needing to compile or install any code.
The SIF file can be downloaded from the
`OSKAR release page <https://github.com/OxfordSKA/OSKAR/releases>`_
and run with the ``singularity exec`` command, which takes the form:

.. code-block:: bash

   singularity exec [flags] <container_path> <app_name> <arguments>...

Use the ``--nv`` flag to enable NVIDIA GPU support in Singularity, if
applicable.

Note also that Singularity will mount the home directory into the container by
default, unless configured otherwise. If you have packages installed in your
home area that should be kept isolated from those in the container (for
example, because of conflicting packages or Python versions, or if you see
other errors caused by trying to load wrong versions of shared libraries when
starting the container) then it may be necessary to disable this either by
using the ``--no-home`` flag, or re-bind the home directory in the container
to somewhere other than your actual $HOME using the ``-H`` flag (for example,
perhaps with ``-H $PWD``).

As an example, to run the ``hello-world.py`` Python script
(from the :ref:`"hello world" example <example_hello_world>`)
and a container image file ``OSKAR-Python3.sif`` (both in the current
directory) on a GPU use:

.. code-block:: bash

   singularity exec --nv ./OSKAR-Python3.sif python3 hello-world.py

Similarly, to run the application ``oskar_sim_interferometer``
with a parameter file ``settings.ini`` in the current directory, use:

.. code-block:: bash

   singularity exec --nv ./OSKAR-Python3.sif oskar_sim_interferometer settings.ini

Running a Singularity container is conceptually similar to running a process,
so this command can be included in a SLURM script and run on a GPU node using
a standard scheduler.
