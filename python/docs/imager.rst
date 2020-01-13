
Imager
======

The :class:`oskar.Imager` class allows basic (dirty) images to be made from
simulated visibility data, either from data files on disk, from numpy arrays
in memory, or from :class:`oskar.VisBlock` objects.

There are three stages required to make an image:

1. Create and set-up an imager.
2. Update the imager with visibility data, possibly multiple times.
3. Finalise the imager to generate the image.

The last two stages can be combined for ease of use if required, simply by
calling the :meth:`run <oskar.Imager.run()>` method with no arguments.

To set up the imager from Python, create an instance of the class and set the
imaging options using a :class:`oskar.SettingsTree` created for
the ``oskar_imager`` application, and/or set properties on the class itself.
These include:

- :meth:`algorithm <oskar.Imager.algorithm>`
- :meth:`cellsize_arcsec <oskar.Imager.cellsize_arcsec>`
- :meth:`channel_snapshots <oskar.Imager.channel_snapshots>`
- :meth:`fft_on_gpu <oskar.Imager.fft_on_gpu>`
- :meth:`fov_deg <oskar.Imager.fov_deg>`
- :meth:`grid_on_gpu <oskar.Imager.grid_on_gpu>`
- :meth:`image_size <oskar.Imager.image_size>`
- :meth:`image_type <oskar.Imager.image_type>`
- :meth:`weighting <oskar.Imager.weighting>`

To optionally filter the input visibility data, use:

- :meth:`freq_max_hz <oskar.Imager.freq_max_hz>`
- :meth:`freq_min_hz <oskar.Imager.freq_min_hz>`
- :meth:`time_max_utc <oskar.Imager.time_max_utc>`
- :meth:`time_min_utc <oskar.Imager.time_min_utc>`
- :meth:`uv_filter_max <oskar.Imager.uv_filter_max>`
- :meth:`uv_filter_min <oskar.Imager.uv_filter_min>`

To specify the input and output files, use:

- :meth:`input_file <oskar.Imager.input_file>`
- :meth:`output_root <oskar.Imager.output_root>`

For convenience, the :meth:`set <oskar.Imager.set>` method can be used to set
multiple properties at once using ``kwargs``.

Off-phase-centre imaging is supported. Use the
:meth:`set_direction <oskar.Imager.set_direction>` method to centre the
image around different coordinates if required.

When imaging visibility data from a file, it is sufficient simply to call
the :meth:`run <oskar.Imager.run>` method with no arguments.
However, it is often necessary to process visibility data prior to imaging it
(perhaps by subtracting model visibilities), and for this reason it may be
more useful to pass the visibility data to the imager explicitly either via
parameters to :meth:`run <oskar.Imager.run>` (which will also finalise
the image) or using the :meth:`update <oskar.Imager.update>` method
(which may be called multiple times if necessary).
The convenience method
:meth:`update_from_block <oskar.Imager.update_from_block>` can be used instead
if visibility data are contained within a :class:`oskar.VisBlock`.

If passing in numpy arrays to :meth:`run <oskar.Imager.run>` or
:meth:`update <oskar.Imager.update>`, be sure to set the frequency and
phase centre first using
:meth:`set_vis_frequency <oskar.Imager.set_vis_frequency>` and
:meth:`set_vis_phase_centre <oskar.Imager.set_vis_phase_centre>`.

After all visibilities have been processed using
:meth:`update <oskar.Imager.update>` or
:meth:`update_from_block <oskar.Imager.update_from_block>`, call
:meth:`finalise <oskar.Imager.finalise>` to generate the image.
The images and/or gridded visibilities can be returned directly to Python as
numpy arrays if required.

Note that uniform weighting requires all visibility coordinates to be known
in advance. To allow for this, set the
:meth:`coords_only <oskar.Imager.coords_only>` property to ``True`` to switch
the imager into a "coordinates-only" mode before calling
:meth:`update <oskar.Imager.update>`. Once all the coordinates have been read,
set :meth:`coords_only <oskar.Imager.coords_only>` to ``False`` and call
:meth:`update <oskar.Imager.update>` again.


Example Usage
-------------

>>> # Generate some data to process.
>>> import numpy
>>> n = 100000  # Number of visibility points.
>>> t = 2 * numpy.pi * numpy.random.random(n)
>>> r = 50e3 * numpy.sqrt(numpy.random.random(n))
>>> # This is for a filled aperture!
>>> uu = r * numpy.cos(t)
>>> vv = r * numpy.sin(t)
>>> ww = numpy.zeros_like(uu)
>>> vis = numpy.ones(n, dtype='c16')  # Single point source at phase centre.

To make an image using supplied (u,v,w) coordinates and visibilities,
and return the image to Python:

>>> # (continued from previous section)
>>> imager = oskar.Imager()
>>> imager.fov_deg = 0.1             # 0.1 degrees across.
>>> imager.image_size = 256          # 256 pixels across.
>>> imager.set_vis_frequency(100e6)  # 100 MHz, single channel data.
>>> imager.update(uu, vv, ww, vis)
>>> data = imager.finalise(return_images=1)
>>> image = data['images'][0]

To plot the image using matplotlib:

>>> # (continued from previous section)
>>> import matplotlib.pyplot as plt
>>> plt.imshow(image)
>>> plt.show()

.. image:: example_image1.png
   :width: 640px
   :align: center
   :height: 480px
   :alt: An example image of a point source, generated using a filled aperture and plotted using matplotlib

An example image of a point source, generated using a filled aperture
and plotted using matplotlib.


Class Methods
-------------

.. autoclass:: oskar.Imager
   :members:
   :special-members: __init__
   :exclude-members: capsule, capsule_ensure, capsule_get, capsule_set,
                     cell, cellsize, cell_size, cell_size_arcsec, fov,
                     input_files, input_vis_data,
                     set_algorithm, set_scale_norm_with_num_input_files,
                     set_cellsize, set_channel_snapshots,
                     set_coords_only,
                     set_fft_on_gpu, set_fov,
                     set_freq_max_hz, set_freq_min_hz,
                     set_grid_on_gpu, set_generate_w_kernels_on_gpu,
                     set_image_size, set_image_type, set_input_file,
                     set_ms_column, set_output_root, set_size,
                     set_time_max_utc, set_time_min_utc,
                     set_uv_filter_max, set_uv_filter_min,
                     set_weighting, set_num_w_planes
