#!/usr/bin/python
import numpy
import oskar
import time
from astropy.io import fits

if __name__ == '__main__':
    # Global options.
    precision = 'single'
    phase_centre_ra_deg = 0.0
    phase_centre_dec_deg = 60.0
    output_root = 'test_via_memory'

    # Define a telescope layout.
    num_stations = 300
    numpy.random.seed(1)
    x = 5000 * numpy.random.randn(num_stations)
    y = 5000 * numpy.random.randn(num_stations)

    # Set up the sky model.
    sky = oskar.Sky.generate_grid(phase_centre_ra_deg, phase_centre_dec_deg,
        16, 1.5, precision=precision)
    sky.append_sources(phase_centre_ra_deg, phase_centre_dec_deg, 1.0)

    # Set up the telescope model.
    tel = oskar.Telescope(precision)
    tel.set_channel_bandwidth(1.0e3)
    tel.set_time_average(10.0)
    tel.set_pol_mode('Scalar')
    tel.set_station_coords_enu(longitude_deg=0, latitude_deg=60, altitude_m=0, 
        x=x, y=y)
    # Set station properties after stations have been defined.
    tel.set_phase_centre(phase_centre_ra_deg, phase_centre_dec_deg)
    tel.set_station_type('Gaussian beam')
    tel.set_gaussian_station_beam_values(5.0, 100e6)

    # Set up two imagers for natural and uniform weighting.
    imagers = []
    for i in range(2):
        imagers.append(oskar.Imager(precision))
        imagers[i].set(fov_deg=2.0, image_size=2048, algorithm='W-projection')
    imagers[0].set(weighting='Natural')
    imagers[1].set(weighting='Uniform')

    # Set up the imaging simulator.
    simulator = oskar.ImagingSimulator(imagers, precision)
    simulator.set_sky_model(sky)
    simulator.set_telescope_model(tel)
    simulator.set_observation_frequency(100e6)
    simulator.set_observation_time(start_time_mjd_utc=51545.0,
        length_sec=43200.0, num_time_steps=48)

    # Simulate and image visibilities.
    start = time.time()
    print('Simulating and imaging...')
    imager_data = simulator.run(return_images=True, return_grids=True)
    print('Completed after %.3f seconds.' % (time.time() - start))

    # Save FITS files (via Python using astropy, not directly from the imager).
    for i in range(len(imagers)):
        # Write the image.
        hdr = fits.header.Header()
        hdr.append(('CTYPE1', 'RA---SIN'))
        hdr.append(('CRVAL1', phase_centre_ra_deg))
        hdr.append(('CRPIX1', imagers[i].image_size / 2 + 1))
        hdr.append(('CDELT1', -imagers[i].cellsize_arcsec / 3600.0))
        hdr.append(('CROTA1', 0.0))
        hdr.append(('CTYPE2', 'DEC--SIN'))
        hdr.append(('CRVAL2', phase_centre_dec_deg))
        hdr.append(('CRPIX2', imagers[i].image_size / 2 + 1))
        hdr.append(('CDELT2', imagers[i].cellsize_arcsec / 3600.0))
        hdr.append(('CROTA2', 0.0))
        hdr.append(('BUNIT', 'Jy/beam'))
        fits.writeto(output_root+'_image_'+imagers[i].weighting+'.fits',
            imager_data[i]['images'][0,:,:], hdr, clobber=True)

        # Write the grid.
        fits.writeto(output_root+'_grid_'+imagers[i].weighting+'.fits',
            numpy.abs(imager_data[i]['grids'][0,:,:]), clobber=True)

