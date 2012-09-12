/*
 * Copyright (c) 2012, The University of Oxford
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

#include "station/oskar_evaluate_station_beam_gaussian.h"
#include "math/cudak/oskar_cudak_gaussian.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_cuda_check_error.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_station_beam_gaussian(oskar_Mem* beam,
        int num_points, const oskar_Mem* l, const oskar_Mem* m,
        double fwhm_deg, int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!beam || !l || !m || !status)
    {
        if (status) *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get type and check consistency. */
    if (beam->type == (OSKAR_DOUBLE | OSKAR_COMPLEX) ||
            l->type == OSKAR_DOUBLE ||
            m->type == OSKAR_DOUBLE)
    {
        type = OSKAR_DOUBLE;
    }
    else if (beam->type == (OSKAR_SINGLE | OSKAR_COMPLEX) ||
            l->type == OSKAR_SINGLE ||
            m->type == OSKAR_SINGLE)
    {
        type = OSKAR_SINGLE;
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Get location and check consistency. */
    location = beam->location;
    if (location != l->location || location != m->location)
        *status = OSKAR_ERR_BAD_LOCATION;

    /* Check that length of input arrays are consistent. */
    if (l->num_elements != num_points || m->num_elements != num_points)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Resize output array if needed. */
    if (beam->num_elements < num_points)
        oskar_mem_realloc(beam, num_points, status);

    /* Check if safe to proceed. */
    if (*status) return;

    double fwhm_rad = fwhm_deg * (M_PI / 180.0);
    double fwhm_lm = sin(fwhm_rad);
    double std = fwhm_lm / (2.0 * sqrt(2.0 * log(2.0)));
    double var = std * std;

    if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            int i;
            for (i = 0; i < num_points; ++i)
            {
                double l_ = ((double*)l->data)[i];
                double m_ = ((double*)m->data)[i];
                double arg = (l_*l_ + m_*m_) / (2.0 * var);
                ((double2*)beam->data)[i].x = exp(-arg);
                ((double2*)beam->data)[i].y = 0.0;
            }
        }
        else /* type == OSKAR_SINGLE */
        {
            int i;
            for (i = 0; i < num_points; ++i)
            {
                float l_ = ((float*)l->data)[i];
                float m_ = ((float*)m->data)[i];
                float arg = (l_*l_ + m_*m_) / (2.0 * var);
                ((float2*)beam->data)[i].x = expf(-arg);
                ((float2*)beam->data)[i].y = 0.0;
            }
        }

    }
    /* GPU version. */
    else
    {
        int num_threads = 256;
        int num_blocks = (num_points + num_threads - 1) / num_threads;
        if (type == OSKAR_DOUBLE)
        {
            oskar_cudak_gaussian_d OSKAR_CUDAK_CONF(num_blocks, num_threads)
            ((double2*)beam->data, num_points, (double*)l->data,
                    (double*)m->data, std);
        }
        else /* type == OSKAR_SINGLE */
        {
            oskar_cudak_gaussian_f OSKAR_CUDAK_CONF(num_blocks, num_threads)
            ((float2*)beam->data, num_points, (float*)l->data,
                    (float*)m->data, std);
        }
        oskar_cuda_check_error(status);
    }
}

#ifdef __cplusplus
}
#endif
