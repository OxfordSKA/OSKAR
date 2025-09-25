/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_STATION_LOAD_SPHERICAL_WAVE_COEFF_FEKO_H5_H_
#define OSKAR_STATION_LOAD_SPHERICAL_WAVE_COEFF_FEKO_H5_H_

/**
 * @file oskar_station_load_spherical_wave_coeff_feko_h5.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads station element spherical wave coefficients from a FEKO HDF5 file.
 *
 * @details
 * This function loads station element spherical wave coefficients from a
 * FEKO HDF5 file.
 *
 * @param[in,out] station Station to update.
 * @param[in] filename    Name of the data file to load.
 * @param[in] max_order   Maximum order of spherical wave coefficients to load.
 *                        A value <= 0 means load all of them.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_station_load_spherical_wave_coeff_feko_h5(
        oskar_Station* station,
        const char* filename,
        int max_order,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
