# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2020, The University of Oxford
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

"""Interfaces to the OSKAR interferometer simulator."""

from __future__ import absolute_import, division, print_function
try:
    from . import _interferometer_lib
except ImportError:
    _interferometer_lib = None
from threading import Thread
from oskar.barrier import Barrier
from oskar.vis_block import VisBlock
from oskar.vis_header import VisHeader

# pylint: disable=useless-object-inheritance
class Interferometer(object):
    """
    This class provides a Python interface to the OSKAR interferometer
    simulator.

    Use the :class:`oskar.Interferometer` class to run interferometry
    simulations from Python using OSKAR.
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

    Examples:

        See the :ref:`example scripts <example_scripts>` section for some
        examples of how to use :class:`oskar.Interferometer`.

    Note:

        It may sometimes be necessary to access the simulated visibility data
        directly as it is generated, instead of loading it afterwards from a
        :class:`Measurement Set <oskar.MeasurementSet>` or
        :class:`file <oskar.Binary>`. This can be significantly more
        efficient than loading and saving visibility data on disk if it needs
        to be modified or processed on-the-fly.

        To do this, create a new class which inherits
        :class:`oskar.Interferometer` and implement a new
        :meth:`process_block() <oskar.Interferometer.process_block()>` method.
        After instantiating it and calling :meth:`run()` on the new subclass,
        the :meth:`process_block()` method will be entered automatically each
        time a new :class:`visibility block <oskar.VisBlock>` has been
        simulated and is ready to process. Use the
        :meth:`vis_header() <oskar.Interferometer.vis_header()>` method to
        obtain access to the visibility data header if required.
        The visibility data and (u,v,w) coordinates can then be accessed or
        manipulated directly using the accessor methods on the
        :class:`oskar.VisHeader` and :class:`oskar.VisBlock` classes.

    """

    def __init__(self, precision=None, settings=None):
        """Creates an OSKAR interferometer simulator.

        Args:
            precision (Optional[str]):
                Either 'double' or 'single' to specify the numerical
                precision of the simulation. Default 'double'.
            settings (Optional[oskar.SettingsTree]):
                Optional settings to use to set up the simulator.
        """
        if _interferometer_lib is None:
            raise RuntimeError("OSKAR library not found.")
        self._capsule = None
        self._barrier = None
        self._settings = None
        if precision is not None and settings is not None:
            raise RuntimeError("Specify either precision or all settings.")
        if precision is None:
            precision = 'double'  # Set default.
        if settings is not None:
            sim = settings.to_interferometer()
            self._capsule = sim.capsule
            self._settings = settings
        self._precision = precision
        self._sky_model_set = False
        self._telescope_model_set = False

    def capsule_ensure(self):
        """Ensures the C capsule exists."""
        if self._capsule is None:
            self._capsule = _interferometer_lib.create(self._precision)

    def capsule_get(self):
        """Returns the C capsule wrapped by the class."""
        return self._capsule

    def capsule_set(self, new_capsule):
        """Sets the C capsule wrapped by the class.

        Args:
            new_capsule (capsule): The new capsule to set.
        """
        if _interferometer_lib.capsule_name(new_capsule) == \
                'oskar_Interferometer':
            del self._capsule
            self._capsule = new_capsule
        else:
            raise RuntimeError("Capsule is not of type oskar_Interferometer.")

    def check_init(self):
        """Initialises the simulator if it has not already been done.

        All options and data must have been set appropriately before calling
        this function.
        """
        self.capsule_ensure()
        if self._settings is not None:
            if not self._sky_model_set:
                self.set_sky_model(self._settings.to_sky())
            if not self._telescope_model_set:
                self.set_telescope_model(self._settings.to_telescope())
        _interferometer_lib.check_init(self._capsule)

    def finalise_block(self, block_index):
        """Finalises a visibility block.

        This method should be called after all prior calls to run_block()
        have completed for a given simulation block.

        Args:
            block_index (int): The simulation block index to finalise.

        Returns:
            block (oskar.VisBlock): A handle to the finalised block.
                This is only valid until the next block is simulated.
        """
        self.capsule_ensure()
        block = VisBlock()
        block.capsule = _interferometer_lib.finalise_block(
            self._capsule, block_index)
        return block

    def finalise(self):
        """Finalises the simulator.

        This method should be called after all blocks have been simulated.
        It is not necessary to call this if using the run() method.
        """
        self.capsule_ensure()
        _interferometer_lib.finalise(self._capsule)

    def get_coords_only(self):
        """Returns whether the simulator provides baseline coordinates only.

        Returns:
            bool: If set, simulate coordinates only.
        """
        self.capsule_ensure()
        return _interferometer_lib.coords_only(self._capsule)

    def get_num_devices(self):
        """Returns the number of compute devices selected.

        Returns:
            int: The number of compute devices selected.
        """
        self.capsule_ensure()
        return _interferometer_lib.num_devices(self._capsule)

    def get_num_gpus(self):
        """Returns the number of GPUs selected.

        Returns:
            int: The number of GPUs selected.
        """
        self.capsule_ensure()
        return _interferometer_lib.num_gpus(self._capsule)

    def get_num_vis_blocks(self):
        """Returns the number of visibility blocks required for the simulation.

        Returns:
            int: The number of visibility blocks required for the simulation.
        """
        self.capsule_ensure()
        return _interferometer_lib.num_vis_blocks(self._capsule)

    def process_block(self, block, block_index):
        """Virtual function to process each visibility block in a worker thread.

        The default implementation simply calls write_block() to write the
        data to any open files. Inherit this class and override this method
        to process the visibilities differently.

        Args:
            block (oskar.VisBlock): A handle to the block to be processed.
            block_index (int):      The index of the visibility block.
        """
        self.capsule_ensure()
        self.write_block(block, block_index)

    def reset_cache(self):
        """Low-level function to reset the simulator's internal memory.
        """
        self.capsule_ensure()
        _interferometer_lib.reset_cache(self._capsule)

    def reset_work_unit_index(self):
        """Low-level function to reset the work unit index.

        This must be called after run_block() has returned, for each block.
        """
        self.capsule_ensure()
        _interferometer_lib.reset_work_unit_index(self._capsule)

    def run_block(self, block_index, device_id=0):
        """Runs the interferometer simulator for one visibility block.

        Multiple compute devices can be used to simulate each block.
        For multi-device simulations, the method should be called multiple
        times using different device IDs from different threads, but with
        the same block index. Device IDs are zero-based.

        This method should be called only after setting all required options.

        Call finalise_block() with the same block index to finalise the block
        after calling this method.

        Args:
            block_index (int): The simulation block index.
            device_id (Optional[int]): The device ID to use for this call.
        """
        self.capsule_ensure()
        _interferometer_lib.run_block(self._capsule, block_index, device_id)

    def run(self):
        """Runs the interferometer simulator.

        The method process_block() is called for each simulated visibility
        block, where a handle to the block is supplied as an argument.
        Inherit this class and override process_block() to process the
        visibilities differently.
        """
        self.capsule_ensure()
        self.check_init()
        self.reset_work_unit_index()
        num_threads = self.num_devices + 1
        self._barrier = Barrier(num_threads)
        threads = []
        for i in range(num_threads):
            threads.append(Thread(target=self._run_blocks, args=[i]))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return self.finalise()

    def set_coords_only(self, value):
        """Sets whether the simulator provides baseline coordinates only.

        Args:
            value (bool): If set, simulate coordinates only.
        """
        self.capsule_ensure()
        _interferometer_lib.set_coords_only(self._capsule, value)

    def set_gpus(self, device_ids):
        """Sets the GPU device IDs to use.

        Args:
            device_ids (int, array-like or None):
                A list of the GPU IDs to use, or -1 to use all.
                If None, then no GPUs will be used.
        """
        self.capsule_ensure()
        _interferometer_lib.set_gpus(self._capsule, device_ids)

    def set_horizon_clip(self, value):
        """Sets whether horizon clipping is performed.

        Args:
            value (bool): If set, apply horizon clipping.
        """
        self.capsule_ensure()
        _interferometer_lib.set_horizon_clip(self._capsule, value)

    def set_max_sources_per_chunk(self, value):
        """Sets the maximum number of sources processed concurrently on one GPU.

        Args:
            value (int): Number of sources per chunk.
        """
        self.capsule_ensure()
        _interferometer_lib.set_max_sources_per_chunk(self._capsule, value)

    def set_max_times_per_block(self, value):
        """Sets the maximum number of times in a visibility block.

        Args:
            value (int): Number of time samples per block.
        """
        self.capsule_ensure()
        _interferometer_lib.set_max_times_per_block(self._capsule, value)

    def set_num_devices(self, value):
        """Sets the number of compute devices to use.

        A compute device may be either a local CPU core, or a GPU.
        To use only a single CPU core for simulation, and no GPUs, call:

        set_gpus(None)
        set_num_devices(1)

        Args:
            value (int): Number of compute devices to use.
        """
        self.capsule_ensure()
        _interferometer_lib.set_num_devices(self._capsule, value)

    def set_observation_frequency(self, start_frequency_hz,
                                  inc_hz=0.0, num_channels=1):
        """Sets observation start frequency, increment, and number of channels.

        Args:
            start_frequency_hz (float): Frequency of the first channel, in Hz.
            inc_hz (Optional[float]): Frequency increment, in Hz.
            num_channels (Optional[int]): Number of frequency channels.
        """
        self.capsule_ensure()
        _interferometer_lib.set_observation_frequency(
            self._capsule, start_frequency_hz, inc_hz, num_channels)

    def set_observation_time(self, start_time_mjd_utc, length_sec,
                             num_time_steps):
        """Sets observation start time, length, and number of samples.

        Args:
            start_time_mjd_utc (float): Observation start time, as MJD(UTC).
            length_sec (float): Observation length in seconds.
            num_time_steps (int): Number of time steps to simulate.
        """
        self.capsule_ensure()
        _interferometer_lib.set_observation_time(
            self._capsule, start_time_mjd_utc, length_sec, num_time_steps)

    def set_output_measurement_set(self, filename):
        """Sets the name of the output CASA Measurement Set.

        Args:
            filename (str): Output filename.
        """
        self.capsule_ensure()
        _interferometer_lib.set_output_measurement_set(self._capsule, filename)

    def set_output_vis_file(self, filename):
        """Sets the name of the output OSKAR visibility file.

        Args:
            filename (str): Output filename.
        """
        self.capsule_ensure()
        _interferometer_lib.set_output_vis_file(self._capsule, filename)

    def set_settings_path(self, filename):
        """Sets the path to the input settings file or script.

        This is used only to store the file in the output visibility data.

        Args:
            filename (str): Filename.
        """
        self.capsule_ensure()
        _interferometer_lib.set_settings_path(self._capsule, filename)

    def set_sky_model(self, sky_model):
        """Sets the sky model used for the simulation.

        Args:
            sky_model (oskar.Sky): Sky model object.
        """
        self.capsule_ensure()
        self._sky_model_set = True
        _interferometer_lib.set_sky_model(self._capsule, sky_model.capsule)

    def set_telescope_model(self, telescope_model):
        """Sets the telescope model used for the simulation.

        Args:
            telescope_model (oskar.Telescope): Telescope model object.
        """
        self.capsule_ensure()
        self._telescope_model_set = True
        _interferometer_lib.set_telescope_model(
            self._capsule, telescope_model.capsule)

    def vis_header(self):
        """Returns the visibility header.

        Returns:
            header (oskar.VisHeader): A handle to the visibility header.
        """
        self.capsule_ensure()
        header = VisHeader()
        header.capsule = _interferometer_lib.vis_header(self._capsule)
        return header

    def write_block(self, block, block_index):
        """Writes a finalised visibility block.

        Args:
            block (oskar.VisBlock): The block to write.
            block_index (int): The simulation block index to write.
        """
        self.capsule_ensure()
        _interferometer_lib.write_block(self._capsule, block.capsule,
                                        block_index)

    # Properties.
    capsule = property(capsule_get, capsule_set)
    coords_only = property(get_coords_only, set_coords_only)
    num_devices = property(get_num_devices, set_num_devices)
    num_gpus = property(get_num_gpus)
    num_vis_blocks = property(get_num_vis_blocks)

    def _run_blocks(self, thread_id):
        """
        Private method to simulate and process visibility blocks concurrently.

        Each thread executes this function.
        For N devices, there will be N+1 threads.
        Thread 0 is used to finalise the block.
        Threads 1 to N (mapped to compute devices) do the simulation.

        Note that no finalisation is performed on the first iteration (as no
        data are ready yet), and no simulation is performed for the last
        iteration (which corresponds to the last block + 1) as this iteration
        simply finalises the last block.

        Args:
            thread_id (int): Zero-based thread ID.
        """
        # Loop over visibility blocks.
        num_blocks = self.num_vis_blocks
        for b in range(num_blocks + 1):
            # Run simulation in threads 1 to N.
            if thread_id > 0 and b < num_blocks:
                self.run_block(b, thread_id - 1)

            # Finalise and process the previous block in thread 0.
            if thread_id == 0 and b > 0:
                block = self.finalise_block(b - 1)
                self.process_block(block, b - 1)

            # Barrier 1: Reset work unit index.
            self._barrier.wait()
            if thread_id == 0:
                self.reset_work_unit_index()

            # Barrier 2: Synchronise before moving to the next block.
            self._barrier.wait()
