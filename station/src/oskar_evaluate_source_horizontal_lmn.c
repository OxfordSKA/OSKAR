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

#include "station/oskar_evaluate_source_horizontal_lmn.h"
#include "sky/oskar_ra_dec_to_hor_lmn_cuda.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_cuda_check_error.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_source_horizontal_lmn(int num_sources, oskar_Mem* l,
        oskar_Mem* m, oskar_Mem* n, const oskar_Mem* RA, const oskar_Mem* Dec,
        const oskar_StationModel* station, double gast, int* status)
{
    double last;

    /* Check all inputs. */
    if (!RA || !Dec || !station || !l || !m || !n || !status)
    {
        if (status) *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Make sure the arrays are on the GPU. */
    if (l->location != OSKAR_LOCATION_GPU ||
            m->location != OSKAR_LOCATION_GPU ||
            n->location != OSKAR_LOCATION_GPU ||
            RA->location != OSKAR_LOCATION_GPU ||
            Dec->location != OSKAR_LOCATION_GPU)
        *status = OSKAR_ERR_BAD_LOCATION;

    /* Check that the dimensions are correct. */
    if (num_sources > RA->num_elements || num_sources > Dec->num_elements)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Resize output arrays if needed. */
    if (l->num_elements < num_sources)
        oskar_mem_realloc(l, num_sources, status);
    if (m->num_elements < num_sources)
        oskar_mem_realloc(m, num_sources, status);
    if (n->num_elements < num_sources)
        oskar_mem_realloc(n, num_sources, status);

    /* Check that the structures contains some sources. */
    if (!RA->data || !Dec->data || !l->data || !m->data || !n->data)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Local Apparent Sidereal Time, in radians. */
    last = gast + station->longitude_rad;

    /* Double precision. */
    if (RA->type == OSKAR_DOUBLE && Dec->type == OSKAR_DOUBLE &&
            l->type == OSKAR_DOUBLE && m->type == OSKAR_DOUBLE &&
            n->type == OSKAR_DOUBLE)
    {
        oskar_ra_dec_to_hor_lmn_cuda_d(num_sources, (double*)(RA->data),
                (double*)(Dec->data), last, station->latitude_rad,
                (double*)(l->data), (double*)(m->data), (double*)(n->data));
        oskar_cuda_check_error(status);
    }

    /* Single precision. */
    else if (RA->type == OSKAR_SINGLE && Dec->type == OSKAR_SINGLE &&
            l->type == OSKAR_SINGLE && m->type == OSKAR_SINGLE &&
            n->type == OSKAR_SINGLE)
    {
        oskar_ra_dec_to_hor_lmn_cuda_f(num_sources, (float*)(RA->data),
                (float*)(Dec->data), (float)last, (float)station->latitude_rad,
                (float*)(l->data), (float*)(m->data), (float*)(n->data));
        oskar_cuda_check_error(status);
    }
    else
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
    }
}

#ifdef __cplusplus
}
#endif
