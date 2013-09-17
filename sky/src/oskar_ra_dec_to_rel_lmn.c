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

#include "sky/oskar_ra_dec_to_rel_lmn.h"
#include "sky/oskar_ra_dec_to_rel_lmn_cuda.h"
#include "math/oskar_sph_to_lm.h"
#include "sky/oskar_lm_to_n.h"
#include <oskar_mem.h>
#include "utility/oskar_cuda_check_error.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_ra_dec_to_rel_lmn_f(int num_points, const float* h_ra,
        const float* h_dec, float ra0_rad, float dec0_rad, float* h_l,
        float* h_m, float* h_n)
{
    /* Compute l,m-direction-cosines of RA, Dec relative to reference point. */
    oskar_sph_to_lm_omp_f(num_points, ra0_rad, dec0_rad, h_ra, h_dec, h_l, h_m);

    /* Compute n-direction-cosines of points from l and m. */
    oskar_lm_to_n_f(num_points, h_l, h_m, h_n);
}

/* Double precision. */
void oskar_ra_dec_to_rel_lmn_d(int num_points, const double* h_ra,
        const double* h_dec, double ra0_rad, double dec0_rad, double* h_l,
        double* h_m, double* h_n)
{
    /* Compute l,m-direction-cosines of RA, Dec relative to reference point. */
    oskar_sph_to_lm_omp_d(num_points, ra0_rad, dec0_rad, h_ra, h_dec, h_l, h_m);

    /* Compute n-direction-cosines of points from l and m. */
    oskar_lm_to_n_d(num_points, h_l, h_m, h_n);
}

/* Wrapper. */
void oskar_ra_dec_to_rel_lmn(int num_points, const oskar_Mem* ra,
        const oskar_Mem* dec, double ra0_rad, double dec0_rad, oskar_Mem* l,
        oskar_Mem* m, oskar_Mem* n,  int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!ra || !dec || !l || !m || !n || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the meta-data. */
    type = oskar_mem_type(ra);
    location = oskar_mem_location(ra);

    /* Check type consistency. */
    if (oskar_mem_type(dec) != type || oskar_mem_type(l) != type || oskar_mem_type(m) != type ||
            oskar_mem_type(n) != type)
        *status = OSKAR_ERR_TYPE_MISMATCH;
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    /* Check location consistency. */
    if (oskar_mem_location(dec) != location || oskar_mem_location(l) != location ||
            oskar_mem_location(m) != location || oskar_mem_location(n) != location)
        *status = OSKAR_ERR_LOCATION_MISMATCH;

    /* Check memory is allocated. */
    if (!ra->data || !dec->data)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check dimensions. */
    if ((int)oskar_mem_length(ra) < num_points || (int)oskar_mem_length(dec) < num_points)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Resize output arrays if needed. */
    if ((int)oskar_mem_length(l) < num_points)
        oskar_mem_realloc(l, num_points, status);
    if ((int)oskar_mem_length(m) < num_points)
        oskar_mem_realloc(m, num_points, status);
    if ((int)oskar_mem_length(n) < num_points)
        oskar_mem_realloc(n, num_points, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Convert coordinates. */
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
        {
            oskar_ra_dec_to_rel_lmn_cuda_f(num_points,
                    (const float*)ra->data, (const float*)dec->data,
                    (float)ra0_rad, (float)dec0_rad, (float*)l->data,
                    (float*)m->data, (float*)n->data);
            oskar_cuda_check_error(status);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_ra_dec_to_rel_lmn_cuda_d(num_points,
                    (const double*)ra->data, (const double*)dec->data,
                    ra0_rad, dec0_rad, (double*)l->data, (double*)m->data,
                    (double*)n->data);
            oskar_cuda_check_error(status);
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_ra_dec_to_rel_lmn_f(num_points,
                    (const float*)ra->data, (const float*)dec->data,
                    (float)ra0_rad, (float)dec0_rad, (float*)l->data,
                    (float*)m->data, (float*)n->data);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_ra_dec_to_rel_lmn_d(num_points,
                    (const double*)ra->data, (const double*)dec->data,
                    ra0_rad, dec0_rad, (double*)l->data, (double*)m->data,
                    (double*)n->data);
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

#ifdef __cplusplus
}
#endif
