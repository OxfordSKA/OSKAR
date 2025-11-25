/*
 * Copyright (c) 2016-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_READ_FITS_IMAGE_PLANE_H_
#define OSKAR_MEM_READ_FITS_IMAGE_PLANE_H_

/**
 * @file oskar_mem_read_fits_image_plane.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Reads pixel data from a FITS image file.
 *
 * @details
 * Reads pixel data from a FITS image file.
 *
 * The returned \p brightness_units string is allocated internally,
 * and must be freed by the caller.
 *
 * @param[in] filename            Name of HEALPix FITS file to read.
 * @param[in] i_time              Zero-based time index of the plane to read.
 * @param[in] i_chan              Zero-based channel index of the plane to read.
 * @param[in] i_stokes            Zero-based Stokes index of the plane to read.
 * @param[out] image_size         Image size[2] (width and height).
 * @param[out] image_crval_deg    Image centre coordinates[2], in degrees.
 * @param[out] image_crpix        Image centre pixels[2] (1-based).
 * @param[out] image_cellsize_deg Image pixel size, in degrees.
 * @param[out] image_time         Time value of the plane.
 * @param[out] image_freq_hz      Frequency value of the plane, in Hz.
 * @param[out] beam_area_pixels   Beam area, in pixels (if found).
 * @param[out] brightness_units   Brightness units (contents of BUNIT keyword).
 * @param[in,out] status          Status return code.
 */
OSKAR_EXPORT
oskar_Mem* oskar_mem_read_fits_image_plane(
        const char* filename,
        int i_time,
        int i_chan,
        int i_stokes,
        int* image_size,
        double* image_crval_deg,
        double* image_crpix,
        double* image_cellsize_deg,
        double* image_time,
        double* image_freq_hz,
        double* beam_area_pixels,
        char** brightness_units,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
