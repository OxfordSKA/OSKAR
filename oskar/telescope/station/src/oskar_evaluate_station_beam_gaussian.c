/*
 * Copyright (c) 2012-2017, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "telescope/station/oskar_evaluate_station_beam_gaussian.h"
#include "telescope/station/oskar_blank_below_horizon.h"
#include "math/oskar_gaussian_circular.h"

#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_station_beam_gaussian(oskar_Mem* beam,
        int num_points, const oskar_Mem* l, const oskar_Mem* m,
        const oskar_Mem* horizon_mask, double fwhm_rad, int* status)
{
    double fwhm_lm, std;
    if (*status) return;

    /* Compute Gaussian standard deviation from FWHM. */
    if (fwhm_rad == 0.0)
    {
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }
    fwhm_lm = sin(fwhm_rad);
    std = fwhm_lm / (2.0 * sqrt(2.0 * log(2.0)));

    /* Evaluate Gaussian. */
    oskar_gaussian_circular(num_points, l, m, std, beam, status);

    /* Blank (zero) sources below the horizon. */
    oskar_blank_below_horizon(beam, horizon_mask, num_points, status);
}

#ifdef __cplusplus
}
#endif
