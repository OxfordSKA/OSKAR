/*
 * Copyright (c) 2011, The University of Oxford
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
#include <cstdlib>
#include <cstdio>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_source_horizontal_lmn(oskar_Mem* l, oskar_Mem* m,
        oskar_Mem* n, const oskar_Mem* RA, const oskar_Mem* Dec,
        const oskar_StationModel* station, const double gast)
{
    if (RA == NULL || Dec == NULL || station == NULL ||
            l == NULL || m == NULL || n == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Make sure the arrays are on the GPU.
    if (l->location() != OSKAR_LOCATION_GPU ||
            m->location() != OSKAR_LOCATION_GPU ||
            n->location() != OSKAR_LOCATION_GPU ||
            RA->location() != OSKAR_LOCATION_GPU ||
            Dec->location() != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    // TODO check arguments properly!
    // Get the number of sources.
//    int num_sources = RA->num_elements();

//    // Check that the dimensions are correct.
//    if (num_sources != Dec->num_elements())
//        return OSKAR_ERR_DIMENSION_MISMATCH;

    // Check that the structures contains some sources.
    if (RA->is_null() || Dec->is_null() ||
            l->is_null() || m->is_null() || n->is_null())
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

//    // Make sure the work arrays are long enough.
//    if (l->num_elements() < num_sources ||
//            m->num_elements() < num_sources ||
//            n->num_elements() < num_sources)
//        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    // Local apparent Sidereal Time, in radians.
    double last = gast + station->longitude_rad;

    // Double precision.
    if (RA->type() == OSKAR_DOUBLE && Dec->type() == OSKAR_DOUBLE &&
            l->type() == OSKAR_DOUBLE && m->type() == OSKAR_DOUBLE &&
            n->type() == OSKAR_DOUBLE)
    {
        return oskar_ra_dec_to_hor_lmn_cuda_d(l->num_elements(), *RA, *Dec,
                last, station->latitude_rad, *l, *m, *n);
    }

    // Single precision.
    else if (RA->type() == OSKAR_SINGLE && Dec->type() == OSKAR_SINGLE &&
            l->type() == OSKAR_SINGLE && m->type() == OSKAR_SINGLE &&
            n->type() == OSKAR_SINGLE)
    {
        return oskar_ra_dec_to_hor_lmn_cuda_f(l->num_elements(), *RA, *Dec,
                (float)last, (float)station->latitude_rad, *l, *m, *n);
    }
    else
    {
        return OSKAR_ERR_TYPE_MISMATCH;
    }
}

#ifdef __cplusplus
}
#endif
