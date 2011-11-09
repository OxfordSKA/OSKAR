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

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_source_horizontal_lmn(oskar_Mem* l, oskar_Mem* m,
        oskar_Mem* n, const oskar_SkyModel* sky,
        const oskar_StationModel* station, const double gast)
{
    if (sky == NULL || station == NULL || l == NULL || m == NULL || n == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Make sure the coordinates in the work and sky arrays are on the GPU.
    if (l->location() != OSKAR_LOCATION_GPU ||
            m->location() != OSKAR_LOCATION_GPU ||
            n->location() != OSKAR_LOCATION_GPU ||
            sky->RA.location() != OSKAR_LOCATION_GPU ||
            sky->Dec.location() != OSKAR_LOCATION_GPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    // Check that the sky structure contains some sources.
    int num_sources = sky->num_sources;
    if (num_sources == 0 || sky->RA.is_null() || sky->Dec.is_null() ||
            l->is_null() || m->is_null() || n->is_null())
    {
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;
    }

    // Make sure the work arrays are long enough.
    if (l->num_elements() != num_sources ||
            m->num_elements() != num_sources ||
            n->num_elements() != num_sources)
    {
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;
    }

    // Local apparent Sidereal Time, in radians.
    double last = gast + station->longitude;

    // Double precision.
    if (sky->type() == OSKAR_DOUBLE && l->type() == OSKAR_DOUBLE &&
            m->type() == OSKAR_DOUBLE && n->type() == OSKAR_DOUBLE)
    {
        return oskar_ra_dec_to_hor_lmn_cuda_d(num_sources, sky->RA, sky->Dec,
                last, station->latitude, *l, *m, *n);
    }

    // Single precision.
    else if (sky->type() == OSKAR_SINGLE && l->type() == OSKAR_SINGLE &&
            m->type() == OSKAR_SINGLE && n->type() == OSKAR_SINGLE)
    {
        return oskar_ra_dec_to_hor_lmn_cuda_f(num_sources, sky->RA, sky->Dec,
                (float)last, (float)station->latitude, *l, *m, *n);
    }
    else
    {
        return OSKAR_ERR_TYPE_MISMATCH;
    }
}

#ifdef __cplusplus
}
#endif
