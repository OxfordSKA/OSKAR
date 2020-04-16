# -*- coding: utf-8 -*-
#
# Copyright (c) 2017-2020, The University of Oxford
# All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#  3. Neither the name of the University of Oxford nor the names of its
#     contributors may be used to endorse or promote products derived from this
#     software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#

"""Interfaces to the OSKAR settings tree."""

from __future__ import absolute_import, print_function
try:
    from . import _settings_lib
    from . import _apps_lib
except ImportError:
    _settings_lib = None
    _apps_lib = None
from oskar.imager import Imager
from oskar.interferometer import Interferometer
from oskar.sky import Sky
from oskar.telescope import Telescope

# pylint: disable=useless-object-inheritance
class SettingsTree(object):
    """This class provides a Python interface to the OSKAR settings tree.

    The :class:`oskar.SettingsTree` class holds parameters used to set up
    other OSKAR Python classes, and is the recommended way of setting up
    :class:`oskar.Interferometer` and :class:`oskar.Imager`.
    It is analogous to the settings files required to run the corresponding
    OSKAR application binaries, and offers a convenient Python interface.
    The constructor requires the name of an OSKAR application binary so that
    it can validate the settings keys, and an exception will be raised if a
    key is not known.

    The complete list of settings keys used by OSKAR applications is
    available in the
    `settings file documentation <https://github.com/OxfordSKA/OSKAR/releases>`_
    and these can also be viewed in the OSKAR GUI.

    After creating an instance of :class:`oskar.SettingsTree`, it can be used
    much like a standard Python dictionary: data access is provided using the
    square-bracket ``[]`` dereference operator using the key name,
    and values can be assigned or read from this. Note that all parameter
    values are stored as strings, and will be converted using Python's
    ``str()`` function as needed.

    Use the :meth:`from_dict() <oskar.SettingsTree.from_dict()>` and
    :meth:`to_dict() <oskar.SettingsTree.to_dict()>` methods to convert
    between a Python dictionary and a :class:`oskar.SettingsTree` if required.

    Examples:

        To set a parameter value:

        .. code-block:: python

           settings = oskar.SettingsTree('oskar_sim_interferometer')
           settings['observation/start_frequency_hz'] = 100e6

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
                   'max_sources_per_chunk': 23000
               },
               'observation' : {
                   'length': 14400.0,
                   'start_frequency_hz': 132e6,
                   'frequency_inc_hz': 100e3,
                   'num_channels': 160,
                   'num_time_steps': 240
               },
               'telescope': {
                   'input_directory': '/path/to/telescope/model',
                   'pol_mode': 'Scalar'
               },
               'interferometer': {
                   'channel_bandwidth_hz': 100e3,
                   'time_average_sec': 1.0,
                   'max_time_samples_per_block': 4
               }
           }
           settings.from_dict(python_dict)


        To get all current parameters (including default values) as a
        Python dictionary:

        .. code-block:: python

           settings = oskar.SettingsTree('oskar_sim_interferometer')
           python_dict = settings.to_dict(include_defaults=True)

    """

    def __init__(self, app=None, settings_file=''):
        """Constructs a settings tree, optionally for the given application.

        If loading a settings file, the application name must also be
        specified.

        Args:
            app (Optional[str]):           Name of the OSKAR application.
            settings_file (Optional[str]): Path of the settings file to load.
        """
        self._capsule = None
        if _settings_lib is None:
            raise RuntimeError("OSKAR library not found.")
        if _apps_lib is None:
            raise RuntimeError("OSKAR apps library not found.")
        if app is None:
            self._capsule = _settings_lib.create()
        else:
            self._capsule = _apps_lib.settings_tree(app, settings_file)

    def from_dict(self, dictionary):
        """Sets the current settings from a Python dictionary.

        Args:
            dictionary (dict):
                Dictionary containing key-value pairs to set.
        """
        return _settings_lib.from_dict(self._capsule, dictionary)

    def set_value(self, key, value, write=True):
        """Sets the value of the setting with the given key.

        Args:
            key (str): Settings key.
            value (str): Settings value.
            write (boolean):
                If true, also write the value to the file. Default True.
        """
        _settings_lib.set_value(self._capsule, key, str(value), write)

    def to_dict(self, include_defaults=False):
        """Returns a Python dictionary containing the current settings.

        Args:
            include_defaults (boolean):
                If true, also return default values.

        Returns:
            dict: Dictionary of key, value pairs.
        """
        return _settings_lib.to_dict(self._capsule, include_defaults)

    def to_imager(self):
        """Returns a new imager from the current settings.

        Returns:
            oskar.Imager: A configured imager.
        """
        imager = Imager()
        imager.capsule = _apps_lib.settings_to_imager(self._capsule)
        return imager

    def to_interferometer(self):
        """Returns a new interferometer simulator from the current settings.

        Returns:
            oskar.Interferometer: A configured interferometer simulator.
        """
        sim = Interferometer()
        sim.capsule = _apps_lib.settings_to_interferometer(self._capsule)
        return sim

    def to_sky(self):
        """Returns a new sky model from the current settings.

        Returns:
            oskar.Sky: A configured sky model.
        """
        sky = Sky()
        sky.capsule = _apps_lib.settings_to_sky(self._capsule)
        return sky

    def to_telescope(self):
        """Returns a new telescope model from the current settings.

        Returns:
            oskar.Telescope: A configured telescope model.
        """
        tel = Telescope()
        tel.capsule = _apps_lib.settings_to_telescope(self._capsule)
        return tel

    def value(self, key):
        """Returns the value of the setting with the given key.

        Args:
            key (str): Settings key.

        Returns:
            str: Value of the setting with the given key.
        """
        return _settings_lib.value(self._capsule, key)

    def __getitem__(self, key):
        """Returns the value of the setting with the given key.

        Args:
            key (str): Settings key.

        Returns:
            str: Value of the setting with the given key.
        """
        return _settings_lib.value(self._capsule, key)

    def __setitem__(self, key, value):
        """Sets the value of the setting with the given key.

        Args:
            key (str): Settings key.
            value (str): Settings value.
        """
        _settings_lib.set_value(self._capsule, key, str(value), True)
