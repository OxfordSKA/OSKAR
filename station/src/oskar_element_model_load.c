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

#include "station/oskar_element_model_load.h"
#include "utility/oskar_getline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEG2RAD 0.0174532925199432957692
#define round(x) ((x)>=0.0?(int)((x)+0.5):(int)((x)-0.5))

int oskar_element_model_load(const char* filename, oskar_ElementModel* data)
{
    // Initialise the flags and local data.
    int n = 0, err = 0;
    float inc_theta = 0.0f, inc_phi = 0.0f, n_theta = 0.0f, n_phi = 0.0f;
    float min_theta = FLT_MAX, max_theta = -FLT_MAX;
    float min_phi = FLT_MAX, max_phi = -FLT_MAX;

    // Declare the line buffer.
    char *line = NULL, *dbi = NULL;
    size_t bufsize = 0;

    // Open the file.
    FILE* file;
    file = fopen(filename, "r");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    // Read the first line and check if data is in logarithmic format.
    err = oskar_getline(&line, &bufsize, file);
    if (err < 0) return err;
    dbi = strstr(line, "dBi"); // Check for presence of "dBi".

    // Initialise pointers to NULL.
    data->g_phi = NULL;
    data->g_theta = NULL;

    // Loop over and read each line in the file.
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        int a;
        float theta = 0.0f, phi = 0.0f, p_theta = 0.0f, p_phi = 0.0f;
        float abs_theta, phase_theta, abs_phi, phase_phi;

        // Parse the line.
        a = sscanf(line, "%f %f %*f %f %f %f %f %*f", &theta, &phi,
                    &abs_theta, &phase_theta, &abs_phi, &phase_phi);

        // Check that data was read correctly.
        if (a != 6) continue;

        // Ignore any data below horizon.
        if (theta > 90.0f) continue;

        // Convert coordinates to radians.
        theta *= DEG2RAD;
        phi *= DEG2RAD;

        // Set coordinate increments.
        if (inc_theta <= FLT_EPSILON) inc_theta = theta - p_theta;
        if (inc_phi <= FLT_EPSILON) inc_phi = phi - p_phi;

        // Set ranges.
        if (theta < min_theta) min_theta = theta;
        if (theta > max_theta) max_theta = theta;
        if (phi < min_phi) min_phi = phi;
        if (phi > max_phi) max_phi = phi;

        // Ensure enough space in arrays.
        if (n % 100 == 0)
        {
            int size;
            size = (n + 100) * sizeof(float);
            data->g_theta = (float2*) realloc(data->g_theta, 2*size);
            data->g_phi   = (float2*) realloc(data->g_phi, 2*size);
        }

        // Store the coordinates in radians.
        p_theta = theta;
        p_phi = phi;

        // Convert decibel to linear scale if necessary.
        if (dbi)
        {
            abs_theta = pow(10.0, abs_theta / 10.0);
            abs_phi   = pow(10.0, abs_phi / 10.0);
        }

        // Amp,phase to real,imag conversion.
        data->g_theta[n].x = abs_theta * cos(phase_theta * DEG2RAD);
        data->g_theta[n].y = abs_theta * sin(phase_theta * DEG2RAD);
        data->g_phi[n].x = abs_phi * cos(phase_phi * DEG2RAD);
        data->g_phi[n].y = abs_phi * sin(phase_phi * DEG2RAD);

        // Increment array pointer.
        n++;
    }

    // Free the line buffer and close the file.
    if (line) free(line);
    fclose(file);

    // Get number of points in each dimension.
    n_theta = (max_theta - min_theta) / inc_theta;
    n_phi = (max_phi - min_phi) / inc_phi;

    // Store number of points in arrays.
    data->n_points = n;
    data->n_theta = 1 + round(n_theta); // Must round to nearest integer.
    data->n_phi = 1 + round(n_phi); // Must round to nearest integer.
    data->min_theta = min_theta;
    data->min_phi = min_phi;
    data->max_theta = max_theta;
    data->max_phi = max_phi;
    data->inc_theta = inc_theta;
    data->inc_phi = inc_phi;

    return 0;
}

#ifdef __cplusplus
}
#endif
