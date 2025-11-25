/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY_H_
#define OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY_H_

/**
 * @file oskar_sky_scale_flux_with_frequency.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Scales all current source brightnesses according to the spectral index for
 * the given frequency.
 *
 * @details
 * This function evaluates all fluxes (all Stokes parameters) at the specified
 * frequency using the spectral index of each source.
 *
 * Frequency scaling is performed as described in the sky model documentation
 * at: https://ska-telescope.gitlab.io/sim/oskar/sky_model/sky_model.html
 *
 * Rotation measure scaling is done as described in the LOFAR BBS page at:
 * https://support.astron.nl/LOFARImagingCookbook/bbs.html#rotation-measure
 *
 * Scaled flux values are written to sky model scratch columns, so the
 * originals remain untouched.
 *
 * @param[in,out] sky      The sky model to re-scale.
 * @param[in] freq_hz      The required frequency, in Hz.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_sky_scale_flux_with_frequency(
        oskar_Sky* sky,
        double freq_hz,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
