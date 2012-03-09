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


#include "math/oskar_sph2cart.h"
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sph2cart(int n, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        oskar_Mem* lon, oskar_Mem* lat)
{
    int i;
    double cosLon, cosLat, sinLon, sinLat;

    if (x == NULL || y == NULL || z == NULL || lon == NULL || lat == NULL)
    {
        return OSKAR_ERR_INVALID_ARGUMENT;
    }

    if (x->location != OSKAR_LOCATION_CPU ||
            y->location != OSKAR_LOCATION_CPU ||
            z->location != OSKAR_LOCATION_CPU ||
            lon->location != OSKAR_LOCATION_CPU ||
            lat->location != OSKAR_LOCATION_CPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    if (x->num_elements > n || y->num_elements > n || z->num_elements > n ||
            lon->num_elements > n || lat->num_elements > n)
    {
        return OSKAR_ERR_OUT_OF_RANGE;
    }

    if (x->type == OSKAR_DOUBLE && y->type == OSKAR_DOUBLE &&
            z->type == OSKAR_DOUBLE && lon->type == OSKAR_DOUBLE &&
            lat->type == OSKAR_DOUBLE)
    {
        for (i = 0; i < n; ++i)
        {
            cosLon = cos(((double*)lon->data)[i]);
            sinLon = sin(((double*)lon->data)[i]);
            cosLat = cos(((double*)lat->data)[i]);
            sinLat = sin(((double*)lat->data)[i]);

            ((double*)x->data)[i] = cosLat * cosLon;
            ((double*)y->data)[i] = cosLat * sinLon;
            ((double*)z->data)[i] = sinLat;
        }
    }
    else if (x->type == OSKAR_SINGLE && y->type == OSKAR_SINGLE &&
            z->type == OSKAR_SINGLE && lon->type == OSKAR_SINGLE &&
            lat->type == OSKAR_SINGLE)
    {
        for (i = 0; i < n; ++i)
        {
            cosLon = cosf(((float*)lon->data)[i]);
            sinLon = sinf(((float*)lon->data)[i]);
            cosLat = cosf(((float*)lat->data)[i]);
            sinLat = sinf(((float*)lat->data)[i]);

            ((float*)x->data)[i] = cosLat * cosLon;
            ((float*)y->data)[i] = cosLat * sinLon;
            ((float*)z->data)[i] = sinLat;
        }
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
