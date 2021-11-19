.. raw:: latex

    \clearpage

Class Reference
===============

Once the OSKAR Python bindings have been installed, use:

.. code-block:: python

   import oskar

in your Python script to access the classes in this package.
The :ref:`example scripts <example_scripts>` may be a helpful place to look
for an overview of what is possible.

All processing-intensive methods on OSKAR Python classes will release the
Global Interpreter Lock (GIL) when called, so that other Python threads may
run concurrently.

Input Data & Settings
---------------------

.. toctree::
   :maxdepth: 2

   settings_tree
   sky
   telescope

Processors
----------

.. toctree::
   :maxdepth: 2

   interferometer
   imager

Visibility Data
---------------

.. toctree::
   :maxdepth: 2

   measurement_set
   binary
   vis_header
   vis_block
