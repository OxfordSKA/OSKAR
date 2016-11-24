/*
 * Copyright (c) 2016, The University of Oxford
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

#include "telescope/oskar_telescope.h"
#include "telescope/private_telescope.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_load_position(oskar_Telescope* telescope,
        const char* filename, int* status)
{
    int num_coords;
    oskar_Mem *lon, *lat, *alt;

    /* Load columns from file. */
    lon = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    lat = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    alt = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    num_coords = (int) oskar_mem_load_ascii(filename, 3, status,
            lon, "", lat, "", alt, "0.0");

    /* Set the telescope centre coordinates. */
    if (num_coords == 1)
    {
        telescope->lon_rad = (oskar_mem_double(lon, status))[0] * M_PI / 180.0;
        telescope->lat_rad = (oskar_mem_double(lat, status))[0] * M_PI / 180.0;
        telescope->alt_metres = (oskar_mem_double(alt, status))[0];
    }
    else
    {
        *status = OSKAR_ERR_BAD_COORD_FILE;
    }

    /* Free memory. */
    oskar_mem_free(lon, status);
    oskar_mem_free(lat, status);
    oskar_mem_free(alt, status);
}

#ifdef __cplusplus
}
#endif
