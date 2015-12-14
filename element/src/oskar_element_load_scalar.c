/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#include <private_element.h>
#include <oskar_element.h>
#include <oskar_getline.h>
#include <oskar_string_to_array.h>
#include <oskar_cmath.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEG2RAD (M_PI/180.0)

static void fit_splines(oskar_Log* log, oskar_Splines* splines, int n,
        oskar_Mem* theta, oskar_Mem* phi, oskar_Mem* data, oskar_Mem* weight,
        double closeness, double closeness_inc, const char* name, int* status);

void oskar_element_load_scalar(oskar_Element* data, oskar_Log* log,
        double freq_hz, const char* filename,
        double closeness, double closeness_inc, int ignore_at_poles,
        int ignore_below_horizon, int* status)
{
    int i, n = 0, type = OSKAR_DOUBLE;
    oskar_Splines *scalar_re = 0, *scalar_im = 0;
    oskar_Mem *theta = 0, *phi = 0, *re = 0, *im = 0, *weight = 0;

    /* Declare the line buffer. */
    char *line = NULL;
    size_t bufsize = 0;
    FILE* file;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check the data type. */
    if (oskar_element_precision(data) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check the location. */
    if (oskar_element_mem_location(data) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check if this frequency has already been set, and get its index if so. */
    n = data->num_freq;
    for (i = 0; i < n; ++i)
    {
        if (fabs(data->freqs_hz[i] - freq_hz) <= freq_hz * DBL_EPSILON)
            break;
    }

    /* Expand arrays to hold data for a new frequency, if needed. */
    if (i >= data->num_freq)
    {
        i = data->num_freq;
        oskar_element_resize_freq_data(data, i + 1, status);
        data->freqs_hz[i] = freq_hz;
    }

    /* Get pointers to surface data based on frequency index. */
    scalar_re = oskar_element_scalar_re(data, i);
    scalar_im = oskar_element_scalar_im(data, i);

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Create local arrays to hold data for fitting. */
    theta  = oskar_mem_create(type, OSKAR_CPU, 0, status);
    phi    = oskar_mem_create(type, OSKAR_CPU, 0, status);
    re     = oskar_mem_create(type, OSKAR_CPU, 0, status);
    im     = oskar_mem_create(type, OSKAR_CPU, 0, status);
    weight = oskar_mem_create(type, OSKAR_CPU, 0, status);
    if (*status) return;

    /* Loop over and read each line in the file. */
    n = 0;
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        double re_, im_;
        double par[] = {0., 0., 0., 0.}; /* theta, phi, amp, phase (optional) */
        void *p_theta = 0, *p_phi = 0, *p_re = 0, *p_im = 0, *p_weight = 0;

        /* Parse the line, and skip if data were not read correctly. */
        if (oskar_string_to_array_d(line, 4, par) < 3)
            continue;

        /* Ignore data below horizon if requested. */
        if (ignore_below_horizon && par[0] > 90.0) continue;

        /* Ignore data at poles if requested. */
        if (ignore_at_poles)
            if (par[0] < 1e-6 || par[0] > (180.0 - 1e-6)) continue;

        /* Convert angular measures to radians. */
        par[0] *= DEG2RAD; /* theta */
        par[1] *= DEG2RAD; /* phi */
        par[3] *= DEG2RAD; /* phase */

        /* Ensure enough space in arrays. */
        if (n % 100 == 0)
        {
            int size;
            size = n + 100;
            oskar_mem_realloc(theta, size, status);
            oskar_mem_realloc(phi, size, status);
            oskar_mem_realloc(re, size, status);
            oskar_mem_realloc(im, size, status);
            oskar_mem_realloc(weight, size, status);
            if (*status) break;
        }
        p_theta  = oskar_mem_void(theta);
        p_phi    = oskar_mem_void(phi);
        p_re     = oskar_mem_void(re);
        p_im     = oskar_mem_void(im);
        p_weight = oskar_mem_void(weight);

        /* Amp,phase to real,imag conversion. */
        re_ = par[2] * cos(par[3]);
        im_ = par[2] * sin(par[3]);

        /* Store the surface data. */
        ((double*)p_theta)[n]  = par[0];
        ((double*)p_phi)[n]    = par[1];
        ((double*)p_re)[n]     = re_;
        ((double*)p_im)[n]     = im_;
        ((double*)p_weight)[n] = 1.0;

        /* Increment array pointer. */
        n++;
    }

    /* Free the line buffer and close the file. */
    free(line);
    fclose(file);

    /* Fit splines to the surface data. */
    fit_splines(log, scalar_re, n, theta, phi, re, weight,
            closeness, closeness_inc, "Scalar [real]", status);
    fit_splines(log, scalar_im, n, theta, phi, im, weight,
            closeness, closeness_inc, "Scalar [imag]", status);

    /* Store the filename. */
    oskar_mem_append_raw(data->filename_scalar[i], filename, OSKAR_CHAR,
                OSKAR_CPU, 1 + strlen(filename), status);

    /* Free local arrays. */
    oskar_mem_free(theta, status);
    oskar_mem_free(phi, status);
    oskar_mem_free(re, status);
    oskar_mem_free(im, status);
    oskar_mem_free(weight, status);
}


static void fit_splines(oskar_Log* log, oskar_Splines* splines, int n,
        oskar_Mem* theta, oskar_Mem* phi, oskar_Mem* data, oskar_Mem* weight,
        double closeness, double closeness_inc, const char* name, int* status)
{
    double avg_frac_error;
    if (*status) return;
    avg_frac_error = closeness; /* Copy the fitting parameter. */
    oskar_log_message(log, 'M', 0, "");
    oskar_log_message(log, 'M', 0, "Fitting surface %s...", name);
    oskar_splines_fit(splines, n, oskar_mem_double(theta, status),
            oskar_mem_double(phi, status), oskar_mem_double_const(data, status),
            oskar_mem_double_const(weight, status), OSKAR_SPLINES_SPHERICAL, 1,
            &avg_frac_error, closeness_inc, 1, 1e-14, status);
    oskar_log_message(log, 'M', 1, "Surface fitted to %.4f average "
            "frac. error (s=%.2e).", avg_frac_error,
            oskar_splines_smoothing_factor(splines));
    oskar_log_message(log, 'M', 1, "Number of knots (theta, phi) = (%d, %d).",
            oskar_splines_num_knots_x_theta(splines),
            oskar_splines_num_knots_y_phi(splines));
}

#ifdef __cplusplus
}
#endif
