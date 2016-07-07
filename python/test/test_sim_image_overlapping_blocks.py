#!/usr/bin/python
from threading import Thread
from oskar import (Sky, Telescope, Simulator, Imager, Barrier)
import logging
import numpy
import os
import time


def run_blocks(simulator, imager, barrier, thread_id):
    """Simulates and images visibility blocks concurrently.

    Each thread executes this function.
    For N devices, there must be N+1 threads.
    Thread 0 is used for imaging on the host.
    Threads 1 to N (mapped to compute devices) do the simulation.

    Note that the imager is not launched on the first loop counter (as no
    data are ready yet), and no simulation is performed for the last loop
    counter (which corresponds to the last block + 1) as this iteration
    finalises and images the last block.

    Args:
        simulator (oskar.Simulator): Simulator object.
        imager (oskar.Imager):       Imager object.
        barrier (oskar.Barrier):     Barrier synchronisation object.
        thread_id (int):             Zero-based thread ID.
    """
    # Loop over visibility blocks.
    num_blocks = simulator.num_vis_blocks()
    for b in range(num_blocks + 1):
        # Run simulation in threads 1 to N.
        if thread_id > 0 and b < num_blocks:
            logging.debug('Simulating block %d/%d', b + 1, num_blocks)
            simulator.run_block(b, thread_id - 1)

        # Run imager in thread 0 for the previous block.
        if thread_id == 0 and b > 0:
            logging.debug('Imaging block %d/%d', b, num_blocks)
            block = simulator.finalise_block(b - 1)
            #simulator.write_block(block, b - 1)
            imager.update_block(block)

        # Barrier 1: Reset work unit index.
        barrier.wait()
        if thread_id == 0:
            simulator.reset_work_unit_index()
            logging.debug('')

        # Barrier 2: Synchronise before moving to the next block.
        barrier.wait()


def run(simulator, imager):
    """Sets up worker threads to simulate and image each block concurrently.

    Args:
        simulator (oskar.Simulator): Simulator object.
        imager (oskar.Imager):       Imager object.
    """
    num_threads = simulator.num_devices() + 1
    barrier = Barrier(num_threads)
    threads = []
    for i in range(num_threads):
        t = Thread(target=run_blocks, args=(simulator, imager, barrier, i))
        threads.append(t)

    # Start all threads and wait for them to finish.
    for t in threads: t.start()
    for t in threads: t.join()


if __name__ == '__main__':
    # Global options.
    logging.basicConfig(level=logging.DEBUG, datefmt='%H:%M:%S',
        format="[%(levelname)-5s] %(asctime)s.%(msecs)03d (%(threadName)-10s) %(message)s")
    precision = 'single'
    phase_centre_ra_deg = 0.0
    phase_centre_dec_deg = 60.0
    start_freq_hz = 100.0e6
    start_time_mjd_utc = 51545.0
    length_sec = 43200.0
    num_time_steps = 48
    inc_sec = length_sec / num_time_steps
    output_root = 'sim_test_overlapping_blocks'

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
    # Set station properties after stations have been defined.    
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

    # Set up the imager.
    imager = Imager(precision)
    imager.set_size(1024)
    imager.set_fov(2.0)
    imager.set_vis_phase_centre(phase_centre_ra_deg, phase_centre_dec_deg)
    imager.set_vis_frequency(start_freq_hz)
    imager.set_vis_time(start_time_mjd_utc, inc_sec, num_time_steps)
    imager.set_output_root(output_root)
    imager.set_algorithm('W-projection')
    imager.set_weighting('Uniform')

    # Loop over all visibility blocks, but generate coordinates only.
    # (Needed for uniform weighting, or W-projection.)
    start = time.time()
    logging.info('Generating coordinates...')
    simulator.set_coords_only(True)
    imager.set_coords_only(True)
    run(simulator, imager)
    simulator.set_coords_only(False)
    imager.set_coords_only(False)

    # Initialise simulator and imager.
    logging.info('Preparing for visibility run...')
    #simulator.set_output_measurement_set(output_root+'.ms')
    simulator.check_init()
    imager.check_init()

    # Loop over all visibility blocks to generate visibilities.
    logging.info('Generating visibilities...')
    run(simulator, imager)
    logging.info('Used %d w-planes.', imager.num_w_planes())

    # Finalise.
    logging.info('Finalising...')
    simulator.finalise()
    imager.finalise()
    logging.info('Completed after %.3f seconds.', time.time() - start)

