/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_SET_SOURCE_H_
#define OSKAR_SKY_SET_SOURCE_H_

/**
 * @file oskar_sky_set_source.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets source data into a sky model.
 *
 * @details
 * This function sets sky model data for a single source at the given index.
 * The sky model must already be large enough to hold the source data.
 *
 * Source data is given in string form.
 *
 * @param[in,out] sky            Pointer to sky model.
 * @param[in] index              Source index in sky model to set.
 * @param[in] str                Source data in string form.
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
void oskar_sky_set_source_str(
        oskar_Sky* sky,
        int index,
        const char* str,
        int* status
);

/**
 * @brief
 * Sets source data into a sky model.
 *
 * @details
 * This function sets sky model data for a single source at the given index.
 * The sky model must already be large enough to hold the source data.
 *
 * Use oskar_sky_set_data() in preference where convenient.
 *
 * @param[in,out] sky            Pointer to sky model.
 * @param[in] index              Source index in sky model to set.
 * @param[in] ra_rad             Source right ascension in radians.
 * @param[in] dec_rad            Source declination in radians.
 * @param[in] stokes_i           Source Stokes I in Jy.
 * @param[in] stokes_q           Source Stokes Q in Jy.
 * @param[in] stokes_u           Source Stokes U in Jy.
 * @param[in] stokes_v           Source Stokes V in Jy.
 * @param[in] ref_frequency_hz   Source reference frequency in Hz.
 * @param[in] spectral_index     Source spectral index.
 * @param[in] rotation_measure   Source rotation measure in radians/m^2.
 * @param[in] fwhm_major_rad     Gaussian source major axis FWHM, in radians.
 * @param[in] fwhm_minor_rad     Gaussian source minor axis FWHM, in radians.
 * @param[in] position_angle_rad Gaussian source position angle, in radians.
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
void oskar_sky_set_source(
        oskar_Sky* sky,
        int index,
        double ra_rad,
        double dec_rad,
        double stokes_i,
        double stokes_q,
        double stokes_u,
        double stokes_v,
        double ref_frequency_hz,
        double spectral_index,
        double rotation_measure,
        double fwhm_major_rad,
        double fwhm_minor_rad,
        double position_angle_rad,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
