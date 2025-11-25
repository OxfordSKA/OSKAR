/*
 * Copyright (c) 2016-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_READ_HEALPIX_FITS_H_
#define OSKAR_MEM_READ_HEALPIX_FITS_H_

/**
 * @file oskar_mem_read_healpix_fits.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Reads pixel data from a HEALPix FITS file.
 *
 * @details
 * Reads pixel data from a HEALPix FITS file.
 *
 * @param[in] filename           Name of HEALPix FITS file to read.
 * @param[in] healpix_hdu_index  Zero-based index of the HEALPix HDU to read.
 * @param[out] nside             HEALPix resolution parameter.
 * @param[out] ordering          HEALPix ordering scheme, either 'R' or 'N'.
 * @param[out] coordsys          'G' for Galactic, 'C' for equatorial.
 * @param[out] brightness_units  Contents of 'TUNIT1' keyword, if present.
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
oskar_Mem* oskar_mem_read_healpix_fits(
        const char* filename,
        int healpix_hdu_index,
        int* nside,
        char* ordering,
        char* coordsys,
        char** brightness_units,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
