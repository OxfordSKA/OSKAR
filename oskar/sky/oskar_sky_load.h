/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_LOAD_H_
#define OSKAR_SKY_LOAD_H_

/**
 * @file oskar_sky_load.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads source data from a plain text source file into an OSKAR sky model
 * structure.
 *
 * @details
 * Sky model files are plain text files that may contain the following 12
 * columns, in order:
 * - RA (deg)
 * - Dec (deg)
 * - Stokes I (Jy)
 * - Stokes Q (Jy)
 * - Stokes U (Jy)
 * - Stokes V (Jy)
 * - Reference frequency at which flux values apply (Hz)
 * - Spectral index
 * - Rotation measure (rad/m^2)
 * - FWHM of Gaussian source major axis (arcseconds)
 * - FWHM of Gaussian source minor axis (arcseconds)
 * - Position angle of Gaussian source major axis (deg)
 *
 * Columns 4 to 12 (Q, U, V, Reference frequency, Spectral index,
 * Rotation measure and extended source parameters) are optional, and default
 * to zero if omitted. Note, however, that a line intended to describe a
 * Gaussian source must explicitly specify all three Gaussian parameters.
 * This means that all columns must be present when using extended sources
 * (but see footnote below).
 *
 * The columns may be space and/or comma separated.
 *
 * Text appearing on a line after a hash symbol (#) is treated as a comment,
 * and is therefore ignored.
 *
 * Footnote:
 * To provide backwards compatibility with old sky model files, a check is made
 * on the number of columns on the line:
 *
 * - Lines containing between 3 and 9 columns set the first 8 parameters
 *   and the rotation measure.
 * - Lines containing 11 columns set the first 8 parameters and the Gaussian
 *   source data (older file format).
 * - Lines containing 12 columns set all source parameters (newer file format).
 * - Lines containing 10 or 13 or more columns set the status flag to
 *   indicate an error, and abort the load.
 *
 * @param[in]  filename  Path to a source list text file.
 * @param[in]  type      Required data type (OSKAR_SINGLE or OSKAR_DOUBLE).
 * @param[in,out] status Status return code.
 *
 * @return A handle to the sky model structure, or NULL if an error occurred.
 */
OSKAR_EXPORT
oskar_Sky* oskar_sky_load(const char* filename, int type, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
