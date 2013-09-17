/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <oskar_evaluate_source_horizontal_lmn.h>
#include <oskar_ra_dec_to_hor_lmn_cuda.h>
#include <oskar_ra_dec_to_hor_lmn.h>
#include <oskar_cuda_check_error.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_source_horizontal_lmn(int num_sources, oskar_Mem* l,
        oskar_Mem* m, oskar_Mem* n, const oskar_Mem* RA, const oskar_Mem* Dec,
        double last, double latitude, int* status)
{
    int type = 0, location;

    /* Check all inputs. */
    if (!RA || !Dec || !l || !m || !n || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that the dimensions are correct. */
    if (num_sources > (int)oskar_mem_length(RA) ||
            num_sources > (int)oskar_mem_length(Dec))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check type. */
    type = oskar_mem_type(RA);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (type != oskar_mem_type(Dec) || type != oskar_mem_type(l) ||
            type != oskar_mem_type(m) || type != oskar_mem_type(n))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check location. */
    location = oskar_mem_location(RA);
    if (location != OSKAR_LOCATION_CPU && location != OSKAR_LOCATION_GPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    if (location != oskar_mem_location(Dec) ||
            location != oskar_mem_location(l) ||
            location != oskar_mem_location(m) ||
            location != oskar_mem_location(n))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Resize output arrays if needed. */
    if ((int)oskar_mem_length(l) < num_sources)
        oskar_mem_realloc(l, num_sources, status);
    if ((int)oskar_mem_length(m) < num_sources)
        oskar_mem_realloc(m, num_sources, status);
    if ((int)oskar_mem_length(n) < num_sources)
        oskar_mem_realloc(n, num_sources, status);
    if (*status) return;

    /* Switch on type and location. */
    if (type == OSKAR_DOUBLE)
    {
        const double *ra_, *dec_;
        double *x_, *y_, *z_;
        ra_ = oskar_mem_double_const(RA, status);
        dec_ = oskar_mem_double_const(Dec, status);
        x_ = oskar_mem_double(l, status);
        y_ = oskar_mem_double(m, status);
        z_ = oskar_mem_double(n, status);

        if (location == OSKAR_LOCATION_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_ra_dec_to_hor_lmn_cuda_d(num_sources, ra_, dec_, last,
                    latitude, x_, y_, z_);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_ra_dec_to_hor_lmn_d(num_sources, ra_, dec_, last,
                    latitude, x_, y_, z_);
        }
    }
    else
    {
        const float *ra_, *dec_;
        float *x_, *y_, *z_;
        ra_ = oskar_mem_float_const(RA, status);
        dec_ = oskar_mem_float_const(Dec, status);
        x_ = oskar_mem_float(l, status);
        y_ = oskar_mem_float(m, status);
        z_ = oskar_mem_float(n, status);

        if (location == OSKAR_LOCATION_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_ra_dec_to_hor_lmn_cuda_f(num_sources, ra_, dec_, (float)last,
                    (float)latitude, x_, y_, z_);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_ra_dec_to_hor_lmn_f(num_sources, ra_, dec_, (float)last,
                    (float)latitude, x_, y_, z_);
        }
    }
}

#ifdef __cplusplus
}
#endif
