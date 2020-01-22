.. _container_singularity:

Singularity
===========

Singularity containers are more suitable for use in traditional multi-user HPC
environments, and are generally easier to use than Docker containers for
running straightforward applications.
Processes in a Singularity container run as the current user so there are fewer
security concerns, and the current directory is mounted by default, so binding
volumes is unnecessary.

For convenience, a SIF-format Singularity container (for use with
Singularity 3.0 and above) can be used to run Python scripts that make use of
OSKAR without needing to compile or install any code.
The SIF file can be downloaded from the
`OSKAR release page <https://github.com/OxfordSKA/OSKAR/releases>`_
and the ``hello-world.py`` Python script
(from the :ref:`"hello world" example <example_hello_world>`)
can be run using:

.. code-block:: bash

   singularity exec --nv <container_path> python3 hello-world.py

where <container_path> is the path to the downloaded container on the
host system, for example:

.. code-block:: bash

   singularity exec --nv ./OSKAR-2.7.6-Python3.sif python3 hello-world.py

Running a Singularity container is conceptually similar to running a process,
so this command can be included in a SLURM script and run on a GPU node using
a standard scheduler.
