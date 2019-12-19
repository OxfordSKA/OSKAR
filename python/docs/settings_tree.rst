
Settings Tree
=============

This class holds parameters used to set up other OSKAR Python classes,
and is the recommended way of setting up :class:`oskar.Interferometer`
and :class:`oskar.Imager`.
It is analogous to the settings files required to run the corresponding
OSKAR application binaries, and offers a convenient Python interface.
The constructor requires the name of an OSKAR application binary so that it
can validate the settings keys, and an exception will be raised if a key is
not known.

The complete list of settings keys used by OSKAR applications is available in
the
`settings file documentation <https://github.com/OxfordSKA/OSKAR/releases>`_
and these can also be viewed in the OSKAR GUI.

After creating an instance of :class:`oskar.SettingsTree`, it can be used
much like a standard Python dictionary: data access is provided using the
square-bracket ``[]`` dereference operator using the key name,
and values can be assigned or read from this. Note that all parameter values
are stored as strings, and should be converted using Python's ``str()``
function if required.

Example Usage
-------------

To set a parameter value (note the explicit string conversion):

.. code-block:: python

   settings = oskar.SettingsTree('oskar_sim_interferometer')
   start_frequency_hz = 100e6
   settings['observation/start_frequency_hz'] = str(start_frequency_hz)

To create a :class:`oskar.SettingsTree` from a Python dictionary
(note that keys can be either flat or nested: for a flat dictionary,
each key must be the fully-qualified name, whereas nested keys are
automatically prefixed by their parent names):

.. code-block:: python

   settings = oskar.SettingsTree('oskar_sim_interferometer')
   python_dict = {
       'simulator': {
           'double_precision': 'true',
           'use_gpus': 'true',
           'max_sources_per_chunk': '23000'
       },
       'observation' : {
           'length': '14400.0',
           'start_frequency_hz': '132e6',
           'frequency_inc_hz': '100e3',
           'num_channels': '160',
           'num_time_steps': '240'
       },
       'telescope': {
           'input_directory': '/path/to/telescope/model',
           'pol_mode': 'Scalar'
       },
       'interferometer': {
           'channel_bandwidth_hz': '100e3',
           'time_average_sec': '1.0',
           'max_time_samples_per_block': '4'
       }
   }
   settings.from_dict(python_dict)


To get all current parameters (including default values) as a
Python dictionary:

.. code-block:: python

   settings = oskar.SettingsTree('oskar_sim_interferometer')
   python_dict = settings.to_dict(include_defaults=True)


Class Methods
-------------

.. autoclass:: oskar.SettingsTree
   :members:
   :special-members: __init__, __getitem__, __setitem__
   :exclude-members: capsule, capsule_ensure, capsule_get, capsule_set,
                     to_imager, to_interferometer, to_sky, to_telescope
