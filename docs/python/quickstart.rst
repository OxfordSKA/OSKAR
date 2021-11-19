.. _python_quickstart:

Quick Start
===========

Those using the Docker, Kubernetes or Singularity container
runtime environments may wish to skip straight to the page on
:ref:`containers`.

The rest of this page describes how to install the OSKAR Python
bindings natively.


Installation
++++++++++++

Linux and macOS
---------------

- Make sure you have a working Python environment
  (including ``pip`` and ``numpy``), C and C++ compilers.

- Make sure OSKAR has been installed.
  On macOS, you can drag the
  `pre-built package <https://github.com/OxfordSKA/OSKAR/releases>`_
  ``OSKAR.app`` to ``/Applications``

- Open a Terminal.

- **(Not usually required)** If OSKAR is installed in a non-standard location,
  edit the paths in ``setup.cfg`` or temporarily set the two environment
  variables:

  .. code-block:: bash

     export OSKAR_INC_DIR=/path/to/oskar/include/folder
     export OSKAR_LIB_DIR=/path/to/oskar/lib

- Install the Python interface with:

  .. code-block:: bash

     pip install --user 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'

- The ``--user`` flag is optional, but you may need root permission without it.

Windows
-------

- Make sure you have a working `Python <https://www.python.org/downloads/windows/>`_
  environment (including ``pip`` and ``numpy``),
  and `Visual Studio Community C and C++ compiler <https://visualstudio.microsoft.com/vs/community/>`_.

  - You will need to make sure that Python is added to the PATH environment
    variable when it is installed.

  - These steps also work with the Anaconda Python distribution,
    but Anaconda is not required.

- Make sure OSKAR has been installed using the `pre-built package <https://github.com/OxfordSKA/OSKAR/releases>`_.

  - In the installer, you will need to select the option **Add OSKAR to the PATH**,
    and install all optional components (headers and libraries).

- Open a Command Prompt (or an Anaconda Prompt, if using Anaconda).

- **(Not usually required)** If OSKAR is installed in a non-standard location,
  edit the paths in ``setup.cfg`` or temporarily set the two environment
  variables:

  .. code-block:: console

     set OSKAR_INC_DIR=C:\path\to\oskar\include\folder
     set OSKAR_LIB_DIR=C:\path\to\oskar\lib

- Install the Python interface with:

  .. code-block:: console

     pip install "git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python"


Using Pipenv
++++++++++++

This works also with `Pipenv <https://docs.pipenv.org>`_
(but make sure the above environment variables are set first, if necessary):

.. code-block:: bash

   pipenv install -e 'git+https://github.com/OxfordSKA/OSKAR.git@master#egg=oskarpy&subdirectory=python'


Uninstallation
++++++++++++++

After installation using the steps above, the OSKAR Python interface can
be uninstalled using:

.. code-block:: bash

   pip uninstall oskarpy

This does not uninstall OSKAR itself, only the Python interface to it.


Usage
+++++

Once the OSKAR Python bindings have been installed, use:

.. code-block:: python

   import oskar

in your Python script to access the classes in this package.
The :ref:`example scripts <example_scripts>` may be helpful.
