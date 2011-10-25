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

#include "apps/lib/oskar_load_telescope.h"
#include "utility/oskar_load_csv_coordinates_2d.h"
#include "interferometry/oskar_horizon_plane_to_itrs.h"
#include <stdlib.h>
#include <stdio.h>

void oskar_load_telescope_d(const char* file_path, const double longitude_rad,
        const double latitude_rad, oskar_TelescopeModel_d* telescope)
{
    double* x_temp = NULL;
    double* y_temp = NULL;
    unsigned n = 0;
    oskar_load_csv_coordinates_2d_d(file_path, &n, &x_temp, &y_temp);

    telescope->latitude  = latitude_rad;
    telescope->longitude = longitude_rad;
    telescope->num_antennas = n;
    size_t mem_size = n * sizeof(double);
    telescope->antenna_x = (double*)malloc(mem_size);
    telescope->antenna_y = (double*)malloc(mem_size);
    telescope->antenna_z = (double*)malloc(mem_size);

    if (telescope->num_antennas == 0)
        fprintf(stderr, "ERROR: no antennas found when loading telescope!\n");

    // Convert horizon x, y coordinates to ITRS (local equatorial system)
    oskar_horizon_plane_to_itrs_d(n, x_temp, y_temp, telescope->latitude,
            telescope->antenna_x, telescope->antenna_y, telescope->antenna_z);

    free(x_temp);
    free(y_temp);
}

void oskar_load_telescope_f(const char* file_path, const float longitude_rad,
        const float latitude_rad, oskar_TelescopeModel_f* telescope)
{
    float* x_temp = NULL;
    float* y_temp = NULL;
    unsigned n = 0;
    oskar_load_csv_coordinates_2d_f(file_path, &n, &x_temp, &y_temp);

    telescope->latitude     = latitude_rad;
    telescope->longitude    = longitude_rad;
    telescope->num_antennas = n;
    size_t mem_size         = n * sizeof(float);
    telescope->antenna_x    = (float*) malloc(mem_size);
    telescope->antenna_y    = (float*) malloc(mem_size);
    telescope->antenna_z    = (float*) malloc(mem_size);

    if (telescope->num_antennas == 0)
        fprintf(stderr, "ERROR: no antennas found when loading telescope!\n");

    // Convert horizon x, y coordinates to ITRS (local equatorial system)
    oskar_horizon_plane_to_itrs_f(n, x_temp, y_temp, telescope->latitude,
            telescope->antenna_x, telescope->antenna_y, telescope->antenna_z);

    free(x_temp);
    free(y_temp);
}
