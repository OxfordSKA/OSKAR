/*
 * Copyright (c) 2012-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/oskar_evaluate_station_beam_gaussian.h"
#include "telescope/station/oskar_blank_below_horizon.h"
#include "math/oskar_gaussian_circular.h"

#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_station_beam_gaussian(int num_points,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* horizon_mask,
        double fwhm_rad, oskar_Mem* beam, int* status)
{
    if (*status) return;

    /* Compute Gaussian standard deviation from FWHM. */
    if (fwhm_rad == 0.0)
    {
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }
    const double fwhm_lm = sin(fwhm_rad);
    const double std = fwhm_lm / (2.0 * sqrt(2.0 * log(2.0)));

    /* Evaluate Gaussian and set points below the horizon to zero. */
    oskar_gaussian_circular(num_points, l, m, std, beam, status);
    oskar_blank_below_horizon(0, num_points, horizon_mask, 0, beam, status);
}

#ifdef __cplusplus
}
#endif
