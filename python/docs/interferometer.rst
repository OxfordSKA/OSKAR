
Interferometer
==============

Use the :class:`oskar.Interferometer` class to run interferometry simulations
from Python using OSKAR.
It requires a model of the sky, a model of the telescope and the observation
parameters as inputs, and it produces a set of simulated visibility data
and (u,v,w) coordinates as outputs.

The most basic way to use this class is as follows:

1. Create a :class:`oskar.SettingsTree` object for the
   ``oskar_sim_interferometer`` application and set required parameters
   either individually or using a Python dictionary.
   These parameters are the same as the ones which appear in the OSKAR GUI.
   (The allowed keys and values are detailed in the
   `settings documentation <https://github.com/OxfordSKA/OSKAR/releases>`_.)

2. Create a :class:`oskar.Interferometer` object and pass it the settings
   via the constructor.

3. Call the :meth:`run() <oskar.Interferometer.run()>` method.

A more flexible way is to partially set the parameters using a
:class:`oskar.SettingsTree` and then override some of them before calling
:meth:`run() <oskar.Interferometer.run()>`.
In particular, the sky model and/or telescope model can be set separately
using the :meth:`set_sky_model() <oskar.Interferometer.set_sky_model()>` and
:meth:`set_telescope_model() <oskar.Interferometer.set_telescope_model()>`
methods, which is useful if some parameters need to be changed as part
of a loop in a script.


Example Usage
-------------


Advanced Usage
--------------

It may sometimes be necessary to access the simulated visibility data directly
as it is generated, instead of loading it afterwards from a
:class:`Measurement Set <oskar.MeasurementSet>` or
:class:`file <oskar.Binary>`. This can be significantly more
efficient than loading and saving visibility data on disk if it needs to be
modified or processed on-the-fly.

To do this, create a new class which inherits :class:`oskar.Interferometer`
and implement a new
:meth:`process_block() <oskar.Interferometer.process_block()>` method.
After instantiating it and calling :meth:`run()` on the new subclass,
the :meth:`process_block()` method will be entered automatically each time
a new :class:`visibility block <oskar.VisBlock>` has been simulated and
is ready to process. Use the
:meth:`vis_header() <oskar.Interferometer.vis_header()>` method to obtain
access to the visibility data header if required.
The visibility data and (u,v,w) coordinates can then be accessed or
manipulated directly using the accessor methods on the
:class:`oskar.VisHeader` and :class:`oskar.VisBlock` classes.


Class Methods
-------------

.. autoclass:: oskar.Interferometer
   :members:
   :special-members: __init__
   :exclude-members: capsule, capsule_ensure, capsule_get, capsule_set,
                     check_init, finalise_block, finalise, get_coords_only,
                     get_num_devices, get_num_gpus, get_num_vis_blocks,
                     reset_work_unit_index, reset_cache, run_block
