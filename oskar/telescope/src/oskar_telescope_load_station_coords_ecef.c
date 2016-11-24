/*
 * Copyright (c) 2013-2016, The University of Oxford
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

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_load_station_coords_ecef(oskar_Telescope* telescope,
        const char* filename, double longitude, double latitude,
        double altitude, int* status)
{
    int num_stations;
    oskar_Mem *x, *y, *z, *x_err, *y_err, *z_err;

    /* Load columns from file into memory. */
    x     = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    y     = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    z     = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    x_err = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    y_err = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    z_err = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    num_stations = (int) oskar_mem_load_ascii(filename, 6, status,
            x, "", y, "", z, "", x_err, "0.0", y_err, "0.0", z_err, "0.0");

    /* Set the station coordinates. */
    oskar_telescope_set_station_coords_ecef(telescope, longitude, latitude,
            altitude, num_stations, x, y, z, x_err, y_err, z_err, status);

    /* Free memory. */
    oskar_mem_free(x, status);
    oskar_mem_free(y, status);
    oskar_mem_free(z, status);
    oskar_mem_free(x_err, status);
    oskar_mem_free(y_err, status);
    oskar_mem_free(z_err, status);
}

#ifdef __cplusplus
}
#endif
