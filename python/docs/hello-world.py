#!/usr/bin/env python3

"""Script to run a simple test example of an OSKAR simulation."""

import matplotlib
matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy
import oskar

# Basic settings. (Note that the sky model is set up later.)
params = {
    "simulator": {
        "use_gpus": False
    },
    "observation" : {
        "num_channels": 3,
        "start_frequency_hz": 100e6,
        "frequency_inc_hz": 20e6,
        "phase_centre_ra_deg": 20,
        "phase_centre_dec_deg": -30,
        "num_time_steps": 24,
        "start_time_utc": "01-01-2000 12:00:00.000",
        "length": "12:00:00.000"
    },
    "telescope": {
        "input_directory": "telescope.tm"
    },
    "interferometer": {
        "oskar_vis_filename": "example.vis",
        "ms_filename": "",
        "channel_bandwidth_hz": 1e6,
        "time_average_sec": 10
    }
}
settings = oskar.SettingsTree("oskar_sim_interferometer")
settings.from_dict(params)

# Set the numerical precision to use.
precision = "single"
if precision == "single":
    settings["simulator/double_precision"] = False

# Create a sky model containing three sources from a numpy array.
sky_data = numpy.array([
        [20.0, -30.0, 1, 0, 0, 0, 100.0e6, -0.7, 0.0, 0,   0,   0],
        [20.0, -30.5, 3, 2, 2, 0, 100.0e6, -0.7, 0.0, 600, 50,  45],
        [20.5, -30.5, 3, 0, 0, 2, 100.0e6, -0.7, 0.0, 700, 10, -10]])
sky = oskar.Sky.from_array(sky_data, precision)  # Pass precision here.

# Set the sky model and run the simulation.
sim = oskar.Interferometer(settings=settings)
sim.set_sky_model(sky)
sim.run()

# Make an image 4 degrees across and return it to Python.
# (It will also be saved with the filename "example_I.fits".)
imager = oskar.Imager(precision)
imager.set(fov_deg=4, image_size=512)
imager.set(input_file="example.vis", output_root="example")
output = imager.run(return_images=1)
image = output["images"][0]

# Render the image using matplotlib and save it as a PNG file.
im = plt.imshow(image, cmap="jet")
plt.gca().invert_yaxis()
plt.colorbar(im)
plt.savefig("%s.png" % imager.output_root)
plt.close("all")
