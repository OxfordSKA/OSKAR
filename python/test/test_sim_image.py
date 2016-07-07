#!/usr/bin/python
from oskar import (Sky, Telescope, Simulator, Imager)
import logging
import numpy
import os
import time

if __name__ == '__main__':
    # Global options.
    logging.basicConfig(level=logging.INFO, datefmt='%H:%M:%S',
        format="[%(levelname)-5s] %(asctime)s.%(msecs)03d %(message)s")
    precision = 'single'
    phase_centre_ra_deg = 0.0
    phase_centre_dec_deg = 60.0
    start_freq_hz = 100.0e6
    start_time_mjd_utc = 51545.0
    length_sec = 43200.0
    num_time_steps = 48
    inc_sec = length_sec / num_time_steps
    output_root = 'sim_test'

    # Define a telescope layout.
    num_stations = 300
    numpy.random.seed(1)
    x = 5000 * numpy.random.randn(num_stations)
    y = 5000 * numpy.random.randn(num_stations)

    # Set up the sky model.
    sky = Sky.generate_grid(precision,
        phase_centre_ra_deg, phase_centre_dec_deg, 16, 1.5)
    sky.append_sources(phase_centre_ra_deg, phase_centre_dec_deg, 1.0)

    # Set up the telescope model.
    tel = Telescope(precision)
    tel.set_channel_bandwidth(1.0e3)
    tel.set_time_average(10.0)
    tel.set_pol_mode('Scalar')
    tel.set_station_coords_enu(longitude_deg=0, latitude_deg=60, altitude_m=0, 
        x=x, y=y)
    # Set phase centre after stations have been defined.
    tel.set_phase_centre(phase_centre_ra_deg, phase_centre_dec_deg)
    tel.set_station_type('Gaussian')
    tel.set_gaussian_station_beam_values(5.0, 100e6)

    # Set up the simulator.
    simulator = Simulator(precision)
    simulator.set_settings_path(os.path.abspath(__file__))
    simulator.set_sky_model(sky)
    simulator.set_telescope_model(tel)
    simulator.set_observation_frequency(start_freq_hz)
    simulator.set_observation_time(start_time_mjd_utc,
        length_sec, num_time_steps)
    simulator.set_gpus(None)
    simulator.set_num_devices(4) # Use 4 CPU threads.
    simulator.set_output_vis_file(output_root+'.vis')
    #simulator.set_output_measurement_set(output_root+'.ms')

    # Set up the imager.
    imager = Imager(precision)
    imager.set_size(1024)
    imager.set_fov(2.0)
    imager.set_input_file(output_root+'.vis')
    imager.set_output_root(output_root)
    imager.set_algorithm('W-projection')
    imager.set_weighting('Uniform')

    # Run the simulator and imager in sequence.
    start = time.time()
    logging.info('Running simulator...')
    simulator.run()
    logging.info('Running imager...')
    imager.run()
    logging.info('Used %d w-planes.' % (imager.num_w_planes()))
    logging.info('Completed after %.3f seconds.', time.time() - start)
