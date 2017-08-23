# -*- coding: utf-8 -*-
"""Example testing the Python OSKAR imager interface."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.io import fits
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import radians, degrees, sin, fabs
from oskar import Imager


def plot_image(ax, image, **imshow_opts):
    """Plot an image with a colorbar in the specified Matplotlib axis.

    Args:
        ax: Matplotlib axis to plot image on.
        image: Image to plot.
        **imshow_opts: Additional options to pass to imshow()
    """
    im = ax.imshow(image, **imshow_opts)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.03)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='small')
    # ticks = np.linspace(image.min(), image.max(), 5)
    # cbar.set_ticks(ticks, update_ticks=True)
    # return cbar


def make_image(uu_m, vv_m, amps, freq_hz, im_size, cell_size_uv_m,
               image_root_name, algorithm='fft'):
    """Make a FITS image from the specified data.

    This function is slightly unusual in that the image size is specified
    using a grid pixel size rather than image pixel size or field-of-view.

    Args:
        uu_m (array_like, float): Baseline uu coordinates, in metres
        vv_m (array_like, float): Baseline vv coordinates, in metres
        amps (array_like, complex): Visibility amplitudes
        freq_hz (float): Observation frequency, in Hz
        im_size (int): Image dimension (assumes square image)
        cell_size_uv_m (float): Image grid pixel separation, in metres
        image_root_name (str): Output FITS image root name.
        algorithm (str): Imaging algorithm to use (FFT or W-projection)

    Returns:
        (dict, str): Dictionary containing the grid used to make the image,
        filename of the output fits image created.
    """
    ra_deg, dec_deg = 0, 90
    ww_m = np.zeros_like(uu_m)
    image_type = 'I'
    wavelength = const.c.value / freq_hz
    uv_cellsize_wavelengths = cell_size_uv_m / wavelength
    fov_rad = Imager.uv_cellsize_to_fov(uv_cellsize_wavelengths, im_size)

    # Create FITS image
    imager = Imager(precision='double')
    imager.set(fov_deg=degrees(fov_rad), size=im_size, algorithm=algorithm,
               weighting='natural', image_type=image_type, wprojplanes=-1,
               output_root=image_root_name)
    imager.set_vis_frequency(freq_hz)
    imager.set_vis_phase_centre(ra_deg, dec_deg)
    imager_data = imager.run(uu_m, vv_m, ww_m, amps, return_grids=1)
    image_grid = imager_data['grids'].squeeze()
    return image_grid, '%s_%s.fits' % (image_root_name, image_type)


def test1():
    """Simple test to check if inverting an image can correctly recover
    the baseline coordinates which the image was created with.
    """
    # Make a FITS image from a freq baseline coordinates
    uu_m = [0.0, 30.0, 50.0]
    vv_m = np.zeros_like(uu_m)
    amps = np.ones_like(uu_m, dtype='c16')
    grid, image_filename = make_image(uu_m, vv_m, amps, freq_hz=const.c.value,
                                      im_size=128, cell_size_uv_m=1,
                                      image_root_name='TEST',
                                      algorithm='FFT')

    # Load created FITS image and extract the image data and header
    # information relating to its coordinates.
    hdu_list = fits.open(image_filename)
    image = hdu_list[0].data.squeeze()
    im_size = hdu_list[0].header['NAXIS1']
    freq_hz = hdu_list[0].header['CRVAL3']
    cell_size_deg = fabs(hdu_list[0].header['CDELT1'])

    # FT the image back to the grid.
    ft_image = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(image)))

    # Get the image and grid coordiante extent (required for plotting)
    wavelength = const.c.value / freq_hz
    fov_deg = degrees(Imager.cellsize_to_fov(radians(cell_size_deg), im_size))
    extent_lm = Imager.image_extent_lm(fov_deg, im_size)
    extent_uv_m = Imager.grid_extent_wavelengths(fov_deg, im_size) * wavelength

    # Plot the image, grid, and FT(image) to check if we have recovered
    # the input positions correctly.
    opts = dict(interpolation='nearest', origin='lower', cmap='gray_r')
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(20, 6), ncols=4)
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.98,
                        wspace=0.3, hspace=0.2)
    plot_image(ax1, image, **dict(opts, extent=extent_lm, vmin=-3, vmax=3))
    ax1.set_title('Image')
    plot_image(ax2, grid.real, **dict(opts, extent=extent_uv_m))
    ax2.set_title('Grid')
    ax2.set_xlabel('uu (m)')
    ax2.set_ylabel('vv (m)')
    plot_image(ax3, ft_image.real, **dict(opts, extent=extent_uv_m))
    ax3.set_title('FT(image).real')
    ax3.set_xlabel('uu (m)')
    ax3.set_ylabel('vv (m)')
    plot_image(ax4, ft_image.imag, **dict(opts, extent=extent_uv_m))
    ax4.set_title('FT(image).imag')
    ax4.set_xlabel('uu (m)')
    ax4.set_ylabel('vv (m)')
    # Add markers for input positions
    marker_opts = dict(radius=3, color='r', fill=None)
    for pos in zip(uu_m, vv_m):
        ax3.add_artist(plt.Circle(pos, **marker_opts))
        ax3.add_artist(plt.Circle(-np.array(pos), **marker_opts))
        ax2.add_artist(plt.Circle(pos, **marker_opts))
    plt.show()


if __name__ == '__main__':
    test1()
