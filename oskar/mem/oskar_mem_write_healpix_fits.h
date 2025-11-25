/*
 * Copyright (c) 2016-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_WRITE_HEALPIX_FITS_H_
#define OSKAR_MEM_WRITE_HEALPIX_FITS_H_

/**
 * @file oskar_mem_write_healpix_fits.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Writes the given array as a HEALPix FITS binary table.
 *
 * @details
 * Writes the given array as a HEALPix FITS binary table.
 *
 * @param[in] data         Array to write.
 * @param[in] filename     Name of HEALPix FITS file to write.
 * @param[in] overwrite    If true, overwrite the file if it exists,
 *                         otherwise append a new FITS binary table.
 * @param[in] nside        HEALPix resolution parameter.
 * @param[in] ordering     'R' for RING, 'N' for NESTED.
 * @param[in] coordsys     'G' for Galactic, 'C' for equatorial.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_mem_write_healpix_fits(
        oskar_Mem* data,
        const char* filename,
        int overwrite,
        int nside,
        char ordering,
        char coordsys,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
