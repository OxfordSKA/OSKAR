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

#include "station/oskar_element_model_load_meerkat.h"
#include "utility/oskar_blank_parentheses.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_string_to_array.h"
#include "utility/oskar_vector_types.h"
#include "math/oskar_SplineData.h"
#include "math/oskar_spline_data_compute_surfit.h"
#include "math/oskar_spline_data_type.h"
#include "math/oskar_spline_data_location.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define round(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

#ifdef __cplusplus
extern "C" {
#endif

#define DEG2RAD 0.0174532925199432957692

int oskar_element_model_load_meerkat(oskar_ElementModel* data, int i,
        int num_files, const char* const* filenames, int search,
        double avg_fractional_err, double s_real, double s_imag)
{
    /* Initialise the flags and local data. */
    int n = 0, err = 0, type = 0, f;
    oskar_SplineData *data_phi = NULL, *data_theta = NULL;
    int k_theta = 0, k_phi = 0;
    double prev_theta = -1.0, prev_phi = -1.0;
    int decimate_theta = 1, decimate_phi = 1; /* Decimation factors. */

    /* Declare the line buffer. */
    char *line = NULL;
    size_t bufsize = 0;
    FILE* file;

    /* Temporary data storage. */
    oskar_Mem m_theta, m_phi, m_theta_re, m_theta_im, m_phi_re, m_phi_im,
    weight;

    /* Get a pointer to the surface to fill. */
    if (i == 1)
    {
        data_phi = &data->port1_phi;
        data_theta = &data->port1_theta;
    }
    else if (i == 2)
    {
        data_phi = &data->port2_phi;
        data_theta = &data->port2_theta;
    }
    else return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check the data types. */
    type = oskar_spline_data_type(data_phi);
    if (type != oskar_spline_data_type(data_theta))
        return OSKAR_ERR_TYPE_MISMATCH;

    /* Check the locations. */
    if (oskar_spline_data_location(data_phi) != OSKAR_LOCATION_CPU ||
            oskar_spline_data_location(data_theta) != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Initialise temporary storage. */
    err = oskar_mem_init(&m_theta, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&m_phi, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&m_theta_re, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&m_theta_im, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&m_phi_re, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&m_phi_im, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
    if (err) return err;
    err = oskar_mem_init(&weight, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
    if (err) return err;

    for (f = 0; f < num_files; ++f)
    {
        /* Open the file. */
        printf("Opening file %s\n", filenames[f]);
        file = fopen(filenames[f], "r");
        if (!file)
            return OSKAR_ERR_FILE_IO;

        /* Read the first four lines. */
        for (i = 0; i < 4; i++)
        {
            err = oskar_getline(&line, &bufsize, file);
            if (err < 0) return err;
        }
        err = 0;

        /* Loop over and read each line in the file. */
        while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
        {
            double data[6];

            /* Parse the line. */
            oskar_blank_parentheses(line);
            if (oskar_string_to_array_d(line, 6, data) != 6) continue;

            /* Ignore any data at poles. */
            if (data[0] < 1e-6 || data[0] > (180.0 - 1e-6)) continue;
            if (data[0] > 8.0) continue;

            /* Keep a record of how many times the coordinates change. */
            if (data[0] != prev_theta)
            {
                prev_theta = data[0];
                k_theta++;
            }
            if (data[1] != prev_phi)
            {
                prev_phi = data[1];
                k_phi++;
            }

            /* Convert data to radians. */
            data[0] *= DEG2RAD;
            data[1] *= DEG2RAD;

            /* Decimate input data. */
            /*
            decimate_phi = 1 + (int)(0.02 / sin(data[0]));
            if (k_theta % decimate_theta) continue;
            if (k_phi % decimate_phi) continue;
            for (i = 2; i < 6; ++i)
            {
                data[i] *= 1024;
                data[i] = (double)(round(data[i]));
                data[i] /= 1024;
            }
            */

            /* Ensure enough space in arrays. */
            if (n % 100 == 0)
            {
                int size;
                size = n + 100;
                err = oskar_mem_realloc(&m_theta, size);
                if (err) return err;
                err = oskar_mem_realloc(&m_phi, size);
                if (err) return err;
                err = oskar_mem_realloc(&m_theta_re, size);
                if (err) return err;
                err = oskar_mem_realloc(&m_theta_im, size);
                if (err) return err;
                err = oskar_mem_realloc(&m_phi_re, size);
                if (err) return err;
                err = oskar_mem_realloc(&m_phi_im, size);
                if (err) return err;
                err = oskar_mem_realloc(&weight, size);
                if (err) return err;
            }

            /* Store the surface data. */
            if (type == OSKAR_SINGLE)
            {
                ((float*)m_theta.data)[n]    = (float)data[0];
                ((float*)m_phi.data)[n]      = (float)data[1];
                ((float*)m_theta_re.data)[n] = (float)data[2];
                ((float*)m_theta_im.data)[n] = (float)data[3];
                ((float*)m_phi_re.data)[n]   = (float)data[4];
                ((float*)m_phi_im.data)[n]   = (float)data[5];
                ((float*)weight.data)[n]     = 1.0;
            }
            else if (type == OSKAR_DOUBLE)
            {
                ((double*)m_theta.data)[n]    = data[0];
                ((double*)m_phi.data)[n]      = data[1];
                ((double*)m_theta_re.data)[n] = data[2];
                ((double*)m_theta_im.data)[n] = data[3];
                ((double*)m_phi_re.data)[n]   = data[4];
                ((double*)m_phi_im.data)[n]   = data[5];
                ((double*)weight.data)[n]     = 1.0;
            }

            /* Increment array pointer. */
            n++;
        }

        /* Close the file. */
        fclose(file);
    }

    /* Free the line buffer. */
    if (line) free(line);

    if (type == OSKAR_SINGLE)
    {
        file = fopen("dump_meerkat.txt", "w");
        for (i = 0; i < n; ++i)
        {
            fprintf(file, "%12.6e, %12.6e, %12.6e, %12.6e, %12.6e, %12.6e, %12.6e\n",
                    ((float*)m_theta.data)[i],
                    ((float*)m_phi.data)[i],
                    ((float*)m_theta_re.data)[i],
                    ((float*)m_theta_im.data)[i],
                    ((float*)m_phi_re.data)[i],
                    ((float*)m_phi_im.data)[i],
                    ((float*)weight.data)[i]);
        }
        fclose(file);
    }

    /* Fit bicubic splines to the surface data. */
    err = oskar_spline_data_compute_surfit(data_theta, n,
            &m_theta, &m_phi, &m_theta_re, &m_theta_im, &weight, &weight,
            search, avg_fractional_err, s_real, s_imag);
    if (err) return err;
    err = oskar_spline_data_compute_surfit(data_phi, n,
            &m_theta, &m_phi, &m_phi_re, &m_phi_im, &weight, &weight,
            search, avg_fractional_err, s_real, s_imag);
    if (err) return err;

    /* Free temporary storage. */
    err = oskar_mem_free(&m_theta);
    if (err) return err;
    err = oskar_mem_free(&m_phi);
    if (err) return err;
    err = oskar_mem_free(&m_theta_re);
    if (err) return err;
    err = oskar_mem_free(&m_theta_im);
    if (err) return err;
    err = oskar_mem_free(&m_phi_re);
    if (err) return err;
    err = oskar_mem_free(&m_phi_im);
    if (err) return err;
    err = oskar_mem_free(&weight);
    if (err) return err;

    return 0;
}

#ifdef __cplusplus
}
#endif
