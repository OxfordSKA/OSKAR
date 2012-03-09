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


#include "math/oskar_sph_rotate_points.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "math/oskar_sph2cart.h"
#include "math/oskar_cart2sph.h"
#include "math/oskar_rotate.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sph_rotate_points(int n, oskar_Mem* lon, oskar_Mem* lat,
        double rot_lon, double rot_lat)
{
    oskar_Mem x, y, z;
    int type, err;

    if (lon == NULL || lat == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (lon->location != OSKAR_LOCATION_CPU || lat->location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    if (lon->num_elements > n || lat->num_elements > n)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    if (lon->type == OSKAR_DOUBLE && lat->type == OSKAR_DOUBLE)
        type = OSKAR_DOUBLE;
    else if (lon->type == OSKAR_SINGLE && lat->type == OSKAR_SINGLE)
        type = OSKAR_SINGLE;
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    err = oskar_mem_init(&x, type, OSKAR_LOCATION_CPU, n, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&y, type, OSKAR_LOCATION_CPU, n, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&z, type, OSKAR_LOCATION_CPU, n, OSKAR_TRUE);
    if (err) return err;

    err = oskar_sph2cart(n, &x, &y, &z, lon, lat);
    if (err) return err;

    err = oskar_rotate_sph(n, &x, &y, &z, rot_lon, rot_lat);
    if (err) return err;

    err = oskar_cart2sph(n, lon, lat, &x, &y, &z);
    if (err) return err;

    oskar_mem_free(&x);
    oskar_mem_free(&y);
    oskar_mem_free(&z);

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
