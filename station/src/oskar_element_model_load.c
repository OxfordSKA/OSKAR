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
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_vector_types.h"
#include "math/oskar_SurfaceData.h"
#include "math/oskar_surface_data_type.h"
#include "math/oskar_surface_data_location.h"
#include "math/oskar_surface_data_resize.h"
#include "math/oskar_surface_data_set_data.h"
#include "math/oskar_surface_data_set_metadata.h"

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
    /* Initialise the flags and local data. */
    int n = 0, err = 0, type = 0;
    double inc_theta = 0.0, inc_phi = 0.0, n_theta = 0.0, n_phi = 0.0;
    double min_theta = DBL_MAX, max_theta = -DBL_MAX;
    double min_phi = DBL_MAX, max_phi = -DBL_MAX;
    oskar_SurfaceData *data_phi = NULL, *data_theta = NULL;

    /* Declare the line buffer. */
    char *line = NULL, *dbi = NULL;
    size_t bufsize = 0;
    FILE* file;

    /* FIXME Get a pointer to the surface to fill. */
    data_phi = &data->port1_phi;
    data_theta = &data->port1_theta;

    /* Check the data types. */
    type = oskar_surface_data_type(data_phi);
    if (type != oskar_surface_data_type(data_theta))
    	return OSKAR_ERR_TYPE_MISMATCH;

    /* Check the locations. */
    if (oskar_surface_data_location(data_phi) != OSKAR_LOCATION_CPU ||
    		oskar_surface_data_location(data_theta) != OSKAR_LOCATION_CPU)
    	return OSKAR_ERR_BAD_LOCATION;

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    /* Read the first line and check if data is in logarithmic format. */
    err = oskar_getline(&line, &bufsize, file);
    if (err < 0) return err;
    err = 0;
    dbi = strstr(line, "dBi"); /* Check for presence of "dBi". */

    /* Loop over and read each line in the file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        int a;
        double theta = 0.0, phi = 0.0, prev_theta = 0.0, prev_phi = 0.0;
        double abs_theta, phase_theta, abs_phi, phase_phi;
        double phi_re, phi_im, theta_re, theta_im;

        /* Parse the line. */
        a = sscanf(line, "%lf %lf %*f %lf %lf %lf %lf %*f", &theta, &phi,
                    &abs_theta, &phase_theta, &abs_phi, &phase_phi);

        /* Check that data was read correctly. */
        if (a != 6) continue;

        /* Ignore any data below horizon. */
        if (theta > 90.0) continue;

        /* Convert data to radians. */
        theta *= DEG2RAD;
        phi *= DEG2RAD;
        phase_theta *= DEG2RAD;
        phase_phi *= DEG2RAD;

        /* Set coordinate increments. */
        if (inc_theta <= DBL_EPSILON) inc_theta = theta - prev_theta;
        if (inc_phi <= DBL_EPSILON) inc_phi = phi - prev_phi;

        /* Set ranges. */
        if (theta < min_theta) min_theta = theta;
        if (theta > max_theta) max_theta = theta;
        if (phi < min_phi) min_phi = phi;
        if (phi > max_phi) max_phi = phi;

        /* Ensure enough space in arrays. */
        if (n % 100 == 0)
        {
            int size;
            size = n + 100;
            err = oskar_surface_data_resize(data_phi, size);
            if (err) return err;
            err = oskar_surface_data_resize(data_theta, size);
            if (err) return err;
        }

        /* Store the coordinates in radians. */
        prev_theta = theta;
        prev_phi = phi;

        /* Convert decibel to linear scale if necessary. */
        if (dbi)
        {
            abs_theta = pow(10.0, abs_theta / 10.0);
            abs_phi   = pow(10.0, abs_phi / 10.0);
        }

        /* Amp,phase to real,imag conversion. */
        theta_re = abs_theta * cos(phase_theta);
        theta_im = abs_theta * sin(phase_theta);
        phi_re = abs_phi * cos(phase_phi);
        phi_im = abs_phi * sin(phase_phi);

        /* Store the surface data. */
        err = oskar_surface_data_set_data(data_phi, n, phi_re, phi_im);
        if (err) return err;
        err = oskar_surface_data_set_data(data_theta, n, theta_re, theta_im);
        if (err) return err;

        /* Increment array pointer. */
        n++;
    }

    /* Free the line buffer and close the file. */
    if (line) free(line);
    fclose(file);

    /* Get number of points in each dimension. */
    n_theta = (max_theta - min_theta) / inc_theta;
    n_phi = (max_phi - min_phi) / inc_phi;

    /* Store number of points in arrays. */
    oskar_surface_data_set_metadata(data_phi,
    		1 + round(n_phi),   /* Must round to nearest integer. */
    		1 + round(n_theta), /* Must round to nearest integer. */
    		inc_phi, inc_theta, min_phi, min_theta, max_phi, max_theta);
    oskar_surface_data_set_metadata(data_theta,
    		1 + round(n_phi),   /* Must round to nearest integer. */
    		1 + round(n_theta), /* Must round to nearest integer. */
    		inc_phi, inc_theta, min_phi, min_theta, max_phi, max_theta);

    return 0;
}

#ifdef __cplusplus
}
#endif
