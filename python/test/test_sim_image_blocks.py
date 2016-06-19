#!/usr/bin/python
from __future__ import print_function
from oskar import (Sky, Telescope, Simulator, Imager, VisBlock)
import logging
import numpy
import os
import time

if __name__ == '__main__':
    # Global options.
    logging.basicConfig(level=logging.DEBUG, 
        format="[%(levelname)s] %(asctime)-15s (%(threadName)-10s) %(message)s")
    precision = 'single'
    phase_centre_ra_deg = 0.0
    phase_centre_dec_deg = 60.0
    start_freq_hz = 100.0e6
    start_time_mjd_utc = 51545.0
    length_sec = 43200.0
    num_time_steps = 48
    inc_sec = length_sec / num_time_steps

    # Define a telescope layout.
    num_stations = 512
    numpy.random.seed(1)
    x = 5000 * numpy.random.randn(num_stations)
    y = 5000 * numpy.random.randn(num_stations)

    # Set up the sky model.
    sky = Sky(precision)
    sky.append_sources(phase_centre_ra_deg, phase_centre_dec_deg, 2.0)

    # Set up the telescope model.
    tel = Telescope(precision)
    tel.set_channel_bandwidth(1.0e3)
    tel.set_time_average(10.0)
    tel.set_pol_mode('Scalar')
    tel.set_station_coords_enu(longitude_deg=0, latitude_deg=60, altitude_m=0, 
        x=x, y=y)
    # Set phase centre after stations have been defined.
    tel.set_phase_centre(phase_centre_ra_deg, phase_centre_dec_deg)

    # Set up the simulator.
    simulator = Simulator(precision)
    simulator.set_settings_path(os.path.abspath(__file__))
    simulator.set_sky_model(sky)
    simulator.set_telescope_model(tel)
    simulator.set_observation_frequency(start_freq_hz)
    simulator.set_observation_time(start_time_mjd_utc,
        length_sec, num_time_steps)
    simulator.set_max_times_per_block(5)
    simulator.check_init()

    # Set up the imager.
    imager = Imager(precision)
    imager.set_size(1024)
    imager.set_fov(2.0)
    imager.set_vis_phase_centre(phase_centre_ra_deg, phase_centre_dec_deg)
    imager.set_vis_frequency(start_freq_hz)
    imager.set_vis_time(start_time_mjd_utc, inc_sec, num_time_steps)
    imager.set_output_root('sim_test_blocks')
    imager.set_algorithm('W-projection')
    imager.set_num_w_planes(256)
    print('Initialising imager...')
    start = time.time()
    imager.check_init()
    print('    Completed after %.3f seconds.' % (time.time() - start))

    # Loop over all blocks.
    print('Looping over blocks in sequence...')
    start = time.time()
    num_blocks = simulator.num_vis_blocks()
    for block_id in range(num_blocks):
        # Simulate the block.
        logging.debug('Simulating block %d/%d', block_id + 1, num_blocks)
        simulator.reset_work_unit_index()
        simulator.run_block(block_id)

        # Image the block.
        logging.debug('Imaging block %d/%d', block_id + 1, num_blocks)
        block_handle = simulator.finalise_block(block_id)
        imager.update_block(block_handle)
    print('    Completed after %.3f seconds.' % (time.time() - start))
    print('    Used %d w-planes.' % (imager.num_w_planes()))

    # Finalise.
    print('Finalising...')
    imager.finalise()
    simulator.finalise()
    print('    Done.')

