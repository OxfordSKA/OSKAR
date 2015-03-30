/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_evaluate_station_beam_gaussian.h>
#include <oskar_blank_below_horizon.h>
#include <oskar_gaussian.h>
#include <oskar_gaussian_cuda.h>
#include <oskar_mem.h>
#include <oskar_cuda_check_error.h>

#include <oskar_cmath.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_station_beam_gaussian(oskar_Mem* beam,
        int num_points, const oskar_Mem* l, const oskar_Mem* m,
        const oskar_Mem* horizon_mask, double fwhm_rad, int* status)
{
    int type, location;
    double fwhm_lm, std;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get type and check consistency. */
    type = oskar_mem_precision(beam);
    if (type != oskar_mem_type(l) || type != oskar_mem_type(m))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (!oskar_mem_is_complex(beam))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    if (fwhm_rad == 0.0)
    {
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }

    /* Get location and check consistency. */
    location = oskar_mem_location(beam);
    if (location != oskar_mem_location(l) ||
            location != oskar_mem_location(m))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check that length of input arrays are consistent. */
    if ((int)oskar_mem_length(l) < num_points ||
            (int)oskar_mem_length(m) < num_points)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Resize output array if needed. */
    if ((int)oskar_mem_length(beam) < num_points)
        oskar_mem_realloc(beam, num_points, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Compute Gaussian standard deviation from FWHM. */
    fwhm_lm = sin(fwhm_rad);
    std = fwhm_lm / (2.0 * sqrt(2.0 * log(2.0)));

    if (type == OSKAR_DOUBLE)
    {
        const double *l_, *m_;
        l_ = oskar_mem_double_const(l, status);
        m_ = oskar_mem_double_const(m, status);

        if (location == OSKAR_CPU)
        {
            if (oskar_mem_is_scalar(beam))
            {
                oskar_gaussian_d(oskar_mem_double2(beam, status),
                        num_points, l_, m_, std);
            }
            else
            {
                oskar_gaussian_md(oskar_mem_double4c(beam, status),
                        num_points, l_, m_, std);
            }
        }
        else
        {
            if (oskar_mem_is_scalar(beam))
            {
                oskar_gaussian_cuda_d(oskar_mem_double2(beam, status),
                        num_points, l_, m_, std);
            }
            else
            {
                oskar_gaussian_cuda_md(oskar_mem_double4c(beam, status),
                        num_points, l_, m_, std);
            }
            oskar_cuda_check_error(status);
        }
    }
    else /* type == OSKAR_SINGLE */
    {
        const float *l_, *m_;
        l_ = oskar_mem_float_const(l, status);
        m_ = oskar_mem_float_const(m, status);

        if (location == OSKAR_CPU)
        {
            if (oskar_mem_is_scalar(beam))
            {
                oskar_gaussian_f(oskar_mem_float2(beam, status), num_points,
                        l_, m_, (float)std);
            }
            else
            {
                oskar_gaussian_mf(oskar_mem_float4c(beam, status), num_points,
                        l_, m_, (float)std);
            }
        }
        else
        {
            if (oskar_mem_is_scalar(beam))
            {
                oskar_gaussian_cuda_f(oskar_mem_float2(beam, status),
                        num_points, l_, m_, (float)std);
            }
            else
            {
                oskar_gaussian_cuda_mf(oskar_mem_float4c(beam, status),
                        num_points, l_, m_, (float)std);
            }
            oskar_cuda_check_error(status);
        }
    }

    /* Blank (zero) sources below the horizon. */
    oskar_blank_below_horizon(beam, horizon_mask, num_points, status);
}

#ifdef __cplusplus
}
#endif
