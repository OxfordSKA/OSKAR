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

#include "station/oskar_element_model_load_cst.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_mem_append_raw.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_vector_types.h"
#include "math/oskar_SettingsSpline.h"
#include "math/oskar_SplineData.h"
#include "math/oskar_spline_data_surfit.h"
#include "math/oskar_spline_data_type.h"
#include "math/oskar_spline_data_location.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG2RAD (M_PI/180.0)

int oskar_element_model_load_cst(oskar_ElementModel* data, int port,
        const char* filename, const oskar_SettingsElementFit* settings)
{
    /* Initialise the flags and local data. */
    int n = 0, err = 0, type = 0, i;
    double cos_overlap;
    oskar_SplineData *data_phi_re = 0, *data_theta_re = 0;
    oskar_SplineData *data_phi_im = 0, *data_theta_im = 0;
    const oskar_SettingsSpline *settings_phi_re = 0, *settings_theta_re = 0;
    const oskar_SettingsSpline *settings_phi_im = 0, *settings_theta_im = 0;

    /* Declare the line buffer. */
    char *line = NULL, *dbi = NULL;
    size_t bufsize = 0;
    FILE* file;

    /* Temporary data storage. */
    oskar_Mem m_theta, m_phi, m_theta_re, m_theta_im, m_phi_re, m_phi_im,
    weight;

    /* Sanity check on inputs. */
    if (port != 1 && port != 2)
        return OSKAR_ERR_INVALID_ARGUMENT;
    if (!data || !filename || !settings)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Get pointers to the surfaces to fill. */
    if (port == 1)
    {
        data_phi_re = &data->port1_phi_re;
        data_phi_im = &data->port1_phi_im;
        data_theta_re = &data->port1_theta_re;
        data_theta_im = &data->port1_theta_im;
    }
    else if (port == 2)
    {
        data_phi_re = &data->port2_phi_re;
        data_phi_im = &data->port2_phi_im;
        data_theta_re = &data->port2_theta_re;
        data_theta_im = &data->port2_theta_im;
    }

    /* Get pointers to the surface fit settings. */
    if (settings->use_common_set)
    {
        settings_phi_re = &settings->all;
        settings_phi_im = &settings->all;
        settings_theta_re = &settings->all;
        settings_theta_im = &settings->all;
    }
    else
    {
        if (port == 1)
        {
            settings_phi_re = &settings->x_phi_re;
            settings_phi_im = &settings->x_phi_im;
            settings_theta_re = &settings->x_theta_re;
            settings_theta_im = &settings->x_theta_im;
        }
        else if (port == 2)
        {
            settings_phi_re = &settings->y_phi_re;
            settings_phi_im = &settings->y_phi_im;
            settings_theta_re = &settings->y_theta_re;
            settings_theta_im = &settings->y_theta_im;
        }
    }

    /* Check the data types. */
    type = oskar_spline_data_type(data_phi_re);
    if (type != oskar_spline_data_type(data_phi_im) ||
            type != oskar_spline_data_type(data_theta_re) ||
            type != oskar_spline_data_type(data_theta_im))
        return OSKAR_ERR_TYPE_MISMATCH;

    /* Check the locations. */
    if (oskar_spline_data_location(data_phi_re) != OSKAR_LOCATION_CPU ||
            oskar_spline_data_location(data_phi_im) != OSKAR_LOCATION_CPU ||
            oskar_spline_data_location(data_theta_re) != OSKAR_LOCATION_CPU ||
            oskar_spline_data_location(data_theta_im) != OSKAR_LOCATION_CPU)
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
        double theta = 0.0, phi = 0.0;
        double abs_theta, phase_theta, abs_phi, phase_phi;
        double phi_re, phi_im, theta_re, theta_im;

        /* Parse the line. */
        a = sscanf(line, "%lf %lf %*f %lf %lf %lf %lf %*f", &theta, &phi,
                    &abs_theta, &phase_theta, &abs_phi, &phase_phi);

        /* Check that data was read correctly. */
        if (a != 6) continue;

        /* Ignore any data below horizon. */
        if (settings->ignore_data_below_horizon && theta > 90.0) continue;

        /* Ignore any data at poles. */
        if (settings->ignore_data_at_pole)
            if (theta < 1e-6 || theta > (180.0 - 1e-6)) continue;

        /* Convert data to radians. */
        theta *= DEG2RAD;
        phi *= DEG2RAD;
        phase_theta *= DEG2RAD;
        phase_phi *= DEG2RAD;

        /* Ensure enough space in arrays. */
        if (n % 100 == 0)
        {
            int size;
            size = n + 100;
            err = oskar_mem_realloc(&m_theta, size);
            if (err) goto cleanup;
            err = oskar_mem_realloc(&m_phi, size);
            if (err) goto cleanup;
            err = oskar_mem_realloc(&m_theta_re, size);
            if (err) goto cleanup;
            err = oskar_mem_realloc(&m_theta_im, size);
            if (err) goto cleanup;
            err = oskar_mem_realloc(&m_phi_re, size);
            if (err) goto cleanup;
            err = oskar_mem_realloc(&m_phi_im, size);
            if (err) goto cleanup;
            err = oskar_mem_realloc(&weight, size);
            if (err) goto cleanup;
        }

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
        if (type == OSKAR_SINGLE)
        {
            ((float*)m_theta.data)[n]    = theta;
            ((float*)m_phi.data)[n]      = phi;
            ((float*)m_theta_re.data)[n] = theta_re;
            ((float*)m_theta_im.data)[n] = theta_im;
            ((float*)m_phi_re.data)[n]   = phi_re;
            ((float*)m_phi_im.data)[n]   = phi_im;
            ((float*)weight.data)[n]     = 1.0;
        }
        else if (type == OSKAR_DOUBLE)
        {
            ((double*)m_theta.data)[n]    = theta;
            ((double*)m_phi.data)[n]      = phi;
            ((double*)m_theta_re.data)[n] = theta_re;
            ((double*)m_theta_im.data)[n] = theta_im;
            ((double*)m_phi_re.data)[n]   = phi_re;
            ((double*)m_phi_im.data)[n]   = phi_im;
            ((double*)weight.data)[n]     = 1.0;
        }

        /* Increment array pointer. */
        n++;
    }

    /* Free the line buffer and close the file. */
    if (line) free(line);
    fclose(file);

    /* Check if any data were loaded. */
    if (n == 0) goto cleanup;

    /* Copy data at phi boundaries. */
    if (type == OSKAR_SINGLE)
    {
        int i;
        i = n - 1;
        while (((float*)m_phi.data)[i] > (2 * M_PI -
                settings->overlap_angle_rad))
        {
            /* Ensure enough space in arrays. */
            if (n % 100 == 0)
            {
                int size;
                size = n + 100;
                err = oskar_mem_realloc(&m_theta, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_theta_re, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_theta_im, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi_re, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi_im, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&weight, size);
                if (err) goto cleanup;
            }
            ((float*)m_theta.data)[n]    = ((float*)m_theta.data)[i];
            ((float*)m_phi.data)[n]      = ((float*)m_phi.data)[i] - 2.0 * M_PI;
            ((float*)m_theta_re.data)[n] = ((float*)m_theta_re.data)[i];
            ((float*)m_theta_im.data)[n] = ((float*)m_theta_im.data)[i];
            ((float*)m_phi_re.data)[n]   = ((float*)m_phi_re.data)[i];
            ((float*)m_phi_im.data)[n]   = ((float*)m_phi_im.data)[i];
            ((float*)weight.data)[n]     = ((float*)weight.data)[i];
            ++n;
            --i;
        }
        i = 0;
        while (((float*)m_phi.data)[i] < settings->overlap_angle_rad)
        {
            /* Ensure enough space in arrays. */
            if (n % 100 == 0)
            {
                int size;
                size = n + 100;
                err = oskar_mem_realloc(&m_theta, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_theta_re, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_theta_im, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi_re, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi_im, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&weight, size);
                if (err) goto cleanup;
            }
            ((float*)m_theta.data)[n]    = ((float*)m_theta.data)[i];
            ((float*)m_phi.data)[n]      = ((float*)m_phi.data)[i] + 2.0 * M_PI;
            ((float*)m_theta_re.data)[n] = ((float*)m_theta_re.data)[i];
            ((float*)m_theta_im.data)[n] = ((float*)m_theta_im.data)[i];
            ((float*)m_phi_re.data)[n]   = ((float*)m_phi_re.data)[i];
            ((float*)m_phi_im.data)[n]   = ((float*)m_phi_im.data)[i];
            ((float*)weight.data)[n]     = ((float*)weight.data)[i];
            ++n;
            ++i;
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        int i;
        i = n - 1;
        while (((double*)m_phi.data)[i] > (2 * M_PI -
                settings->overlap_angle_rad))
        {
            /* Ensure enough space in arrays. */
            if (n % 100 == 0)
            {
                int size;
                size = n + 100;
                err = oskar_mem_realloc(&m_theta, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_theta_re, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_theta_im, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi_re, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi_im, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&weight, size);
                if (err) goto cleanup;
            }
            ((double*)m_theta.data)[n]    = ((double*)m_theta.data)[i];
            ((double*)m_phi.data)[n]      = ((double*)m_phi.data)[i] - 2.0 * M_PI;
            ((double*)m_theta_re.data)[n] = ((double*)m_theta_re.data)[i];
            ((double*)m_theta_im.data)[n] = ((double*)m_theta_im.data)[i];
            ((double*)m_phi_re.data)[n]   = ((double*)m_phi_re.data)[i];
            ((double*)m_phi_im.data)[n]   = ((double*)m_phi_im.data)[i];
            ((double*)weight.data)[n]     = ((double*)weight.data)[i];
            ++n;
            --i;
        }
        i = 0;
        while (((double*)m_phi.data)[i] < settings->overlap_angle_rad)
        {
            /* Ensure enough space in arrays. */
            if (n % 100 == 0)
            {
                int size;
                size = n + 100;
                err = oskar_mem_realloc(&m_theta, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_theta_re, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_theta_im, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi_re, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&m_phi_im, size);
                if (err) goto cleanup;
                err = oskar_mem_realloc(&weight, size);
                if (err) goto cleanup;
            }
            ((double*)m_theta.data)[n]    = ((double*)m_theta.data)[i];
            ((double*)m_phi.data)[n]      = ((double*)m_phi.data)[i] + 2.0 * M_PI;
            ((double*)m_theta_re.data)[n] = ((double*)m_theta_re.data)[i];
            ((double*)m_theta_im.data)[n] = ((double*)m_theta_im.data)[i];
            ((double*)m_phi_re.data)[n]   = ((double*)m_phi_re.data)[i];
            ((double*)m_phi_im.data)[n]   = ((double*)m_phi_im.data)[i];
            ((double*)weight.data)[n]     = ((double*)weight.data)[i];
            ++n;
            ++i;
        }
    }

    /* Re-weight at boundaries and overlap region of phi. */
    cos_overlap = cos(settings->overlap_angle_rad);
    if (type == OSKAR_SINGLE)
    {
        for (i = 0; i < n; ++i)
        {
            if (fabs(cos(((float*)m_phi.data)[i]) - 1.0) < 0.001)
                ((float*)weight.data)[i] = settings->weight_boundaries;
            else if (cos(((float*)m_phi.data)[i]) > cos_overlap)
                ((float*)weight.data)[i] = settings->weight_overlap;
        }
    }
    else
    {
        for (i = 0; i < n; ++i)
        {
            if (fabs(cos(((double*)m_phi.data)[i]) - 1.0) < 0.001)
                ((double*)weight.data)[i] = settings->weight_boundaries;
            else if (cos(((double*)m_phi.data)[i]) > cos_overlap)
                ((double*)weight.data)[i] = settings->weight_overlap;
        }
    }

#if 0
    if (type == OSKAR_SINGLE)
    {
        file = fopen("dump.txt", "w");
        for (i = 0; i < n; ++i)
        {
            fprintf(file, "%9.4f, %9.4f, %9.4f, %9.4f, %9.4f, %9.4f, %9.4f\n",
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
    else if (type == OSKAR_DOUBLE)
    {
        file = fopen("dump.txt", "w");
        for (i = 0; i < n; ++i)
        {
            fprintf(file, "%9.4f, %9.4f, %9.4f, %9.4f, %9.4f, %9.4f, %9.4f\n",
                    ((double*)m_theta.data)[i],
                    ((double*)m_phi.data)[i],
                    ((double*)m_theta_re.data)[i],
                    ((double*)m_theta_im.data)[i],
                    ((double*)m_phi_re.data)[i],
                    ((double*)m_phi_im.data)[i],
                    ((double*)weight.data)[i]);
        }
        fclose(file);
    }
#endif

    /* Fit bicubic splines to the surface data. */
    err = oskar_spline_data_surfit(data_theta_re, n,
            &m_theta, &m_phi, &m_theta_re, &weight, settings_theta_re);
    if (err) goto cleanup;
    err = oskar_spline_data_surfit(data_theta_im, n,
            &m_theta, &m_phi, &m_theta_im, &weight, settings_theta_im);
    if (err) goto cleanup;
    err = oskar_spline_data_surfit(data_phi_re, n,
            &m_theta, &m_phi, &m_phi_re, &weight, settings_phi_re);
    if (err) goto cleanup;
    err = oskar_spline_data_surfit(data_phi_im, n,
            &m_theta, &m_phi, &m_phi_im, &weight, settings_phi_im);
    if (err) goto cleanup;

    /* Store the filename. */
    if (port == 1)
    {
        err = oskar_mem_init(&data->filename_port1, OSKAR_CHAR,
                OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
        if (err) return err;
        err = oskar_mem_append_raw(&data->filename_port1, filename,
                OSKAR_CHAR, OSKAR_LOCATION_CPU, 1 + strlen(filename));
        if (err) return err;
    }
    else if (port == 2)
    {
        err = oskar_mem_init(&data->filename_port2, OSKAR_CHAR,
                OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
        if (err) return err;
        err = oskar_mem_append_raw(&data->filename_port2, filename,
                OSKAR_CHAR, OSKAR_LOCATION_CPU, 1 + strlen(filename));
        if (err) return err;
    }

    /* Free temporary storage. */
    cleanup:
    oskar_mem_free(&m_theta);
    oskar_mem_free(&m_phi);
    oskar_mem_free(&m_theta_re);
    oskar_mem_free(&m_theta_im);
    oskar_mem_free(&m_phi_re);
    oskar_mem_free(&m_phi_im);
    oskar_mem_free(&weight);

    return err;
}

#ifdef __cplusplus
}
#endif
