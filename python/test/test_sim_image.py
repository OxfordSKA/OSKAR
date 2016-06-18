#!/usr/bin/python
from __future__ import print_function
from oskar import (Sky, Telescope, Simulator, Imager)
import numpy
import time

if __name__ == '__main__':
    # Global options.
    precision = 'single'
    phase_centre_ra_deg = 0.0
    phase_centre_dec_deg = 60.0
    start_freq_hz = 100.0e6
    start_time_mjd_utc = 51545.0
    length_sec = 43200.0
    num_time_steps = 48
    inc_sec = length_sec / num_time_steps
    vis_file = 'sim_test.vis'

    # Define a telescope layout.
    num_stations = 512
    numpy.random.seed(1)
    x = 5000 * numpy.random.randn(num_stations)
    y = 5000 * numpy.random.randn(num_stations)

    # Set up the sky model.
    sky = Sky(precision)
    sky.append_data(phase_centre_ra_deg, phase_centre_dec_deg, 2.0)

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
    simulator.set_sky_model(sky)
    simulator.set_telescope_model(tel)
    simulator.set_observation_frequency(start_freq_hz)
    simulator.set_observation_time(start_time_mjd_utc,
        length_sec, num_time_steps)
    simulator.set_gpus(-1)
    simulator.set_output_measurement_set(vis_file+'.ms')

    # Set up the imager.
    imager = Imager(precision)
    imager.set_size(1024)
    imager.set_fov(2.0)
    imager.set_input_file(vis_file+'.ms')
    imager.set_output_root('sim_test')
    imager.set_algorithm('W-projection')
    imager.set_num_w_planes(256)

    # Run the simulator and imager in sequence.
    print('Running simulator...')
    start = time.time()
    simulator.run()
    print('    Completed after %.3f seconds.' % (time.time() - start))
    print('Running imager...')
    start = time.time()
    imager.run()
    print('    Used %d w-planes.' % (imager.num_w_planes()))
    print('    Completed after %.3f seconds.' % (time.time() - start))
