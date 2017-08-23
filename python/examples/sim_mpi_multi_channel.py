#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import oskar
from mpi4py import MPI

if __name__ == '__main__':
    # Check command line arguments.
    if len(sys.argv) < 3:
        raise RuntimeError(
            'Usage: mpiexec -n <np> '
            'python sim_mpi_multi_channel.py '
            '<freq_start_MHz> <freq_inc_MHz>')

    # Global options.
    precision = 'single'
    phase_centre_ra_deg = 0.0
    phase_centre_dec_deg = -60.0

    # Get MPI communicator and rank, and set values that depend on the rank.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    frequency_start_hz = 1e6 * float(sys.argv[-2])
    frequency_inc_hz = 1e6 * float(sys.argv[-1])
    frequency_hz = frequency_start_hz + frequency_inc_hz * rank
    filename = ('test_data_%05.1f.ms' % (frequency_hz / 1e6))

    # Set up the sky model.
    sky = oskar.Sky.generate_grid(phase_centre_ra_deg, phase_centre_dec_deg,
                                  16, 5, precision=precision)
    sky.append_sources(phase_centre_ra_deg, phase_centre_dec_deg, 1.0)

    # Set up the telescope model.
    tel = oskar.Telescope(precision)
    tel.set_channel_bandwidth(100e3)
    tel.set_time_average(10.0)
    tel.set_pol_mode('Scalar')
    tel.load('SKA1-LOW_v5_single_random.tm')
    # Set station properties after stations have been defined.
    tel.set_phase_centre(phase_centre_ra_deg, phase_centre_dec_deg)
    tel.set_station_type('Isotropic')

    # Set up the basic simulator and run simulation.
    simulator = oskar.Interferometer(precision)
    simulator.set_settings_path(os.path.abspath(__file__))
    simulator.set_max_sources_per_chunk(sky.num_sources+1)
    simulator.set_sky_model(sky)
    simulator.set_telescope_model(tel)
    simulator.set_observation_frequency(frequency_hz)
    simulator.set_observation_time(
        start_time_mjd_utc=51544.375, length_sec=10800.0, num_time_steps=180)
    simulator.set_output_measurement_set(filename)
    start = time.time()
    simulator.run()
    print('Simulation for %05.1f MHz completed after %.3f seconds.' %
          (frequency_hz / 1e6, time.time() - start))
