
Telescope Model
===============

A telescope model contains the parameters and data used to describe a
telescope configuration, primarily station and element coordinate data.

Telescope models used by OSKAR are stored in a directory structure, which is
fully described in the
`telescope model documentation <https://github.com/OxfordSKA/OSKAR/releases>`_.
The :class:`oskar.Telescope` class can be used to load a telescope model
(using the :meth:`load() <oskar.Telescope.load()>` method) and, to a limited
extent, override parts of it once it has been loaded: however, most users
will probably wish to simply load a new telescope model.

**Note that the** :class:`oskar.Telescope` **class does not need to be used**
if it is sufficient to use only the :class:`oskar.SettingsTree` to set up the
simulations. It may be sufficient to ignore this page!

If required, element-level data can be overridden (in memory only) once the
telescope model has been loaded using the methods:

- :meth:`override_element_cable_length_errors() <oskar.Telescope.override_element_cable_length_errors()>`
- :meth:`override_element_gains() <oskar.Telescope.override_element_gains()>`
- :meth:`override_element_phases() <oskar.Telescope.override_element_phases()>`

The phase centre of the telescope should be set using the
:meth:`set_phase_centre() <oskar.Telescope.set_phase_centre()>` method.

Example Usage
-------------

To load a telescope model stored in the directory ``telescope.tm``:

>>> tel = oskar.Telescope()
>>> tel.load('telescope.tm')


Class Methods
-------------

.. autoclass:: oskar.Telescope
   :members:
   :special-members: __init__
   :exclude-members: capsule, capsule_ensure, capsule_get, capsule_set,
                     get_identical_stations, get_max_station_depth,
                     get_max_station_size, get_num_baselines, get_num_stations,
                     set_channel_bandwidth, set_time_average, set_uv_filter
