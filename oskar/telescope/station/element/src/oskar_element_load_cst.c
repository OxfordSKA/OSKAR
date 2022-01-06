/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "log/oskar_log.h"
#include "math/oskar_cmath.h"
#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"
#include "utility/oskar_getline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEG2RAD (M_PI/180.0)

static void fit_splines(oskar_Splines** splines_ptr, int n,
        oskar_Mem* theta, oskar_Mem* phi, oskar_Mem* data, oskar_Mem* weight,
        double closeness, double closeness_inc, const char* name,
        oskar_Log* log, int* status);

void oskar_element_load_cst(oskar_Element* data,
        int port, double freq_hz, const char* filename,
        double closeness, double closeness_inc, int ignore_at_poles,
        int ignore_below_horizon, oskar_Log* log, int* status)
{
    int i = 0, n = 0;
    oskar_Splines **data_h_re = 0, **data_h_im = 0;
    oskar_Splines **data_v_re = 0, **data_v_im = 0;
    oskar_Mem *theta = 0, *phi = 0;
    oskar_Mem *h_re = 0, *h_im = 0, *v_re = 0, *v_im = 0, *weight = 0;

    /* Declare the line buffer. */
    char *line = 0, *dbi = 0, *ludwig3 = 0;
    size_t bufsize = 0;
    FILE* file = 0;

    /* Check inputs. */
    if (*status) return;
    if (port != 0 && port != 1 && port != 2)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    if (oskar_element_precision(data) != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
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
        {
            break;
        }
    }

    /* Expand arrays to hold data for a new frequency, if needed. */
    if (i >= data->num_freq)
    {
        i = data->num_freq;
        oskar_element_resize_freq_data(data, i + 1, status);
        data->freqs_hz[i] = freq_hz;
    }

    /* Get pointers to surface data based on port number and frequency index. */
    if (port == 1 || port == 0)
    {
        data_h_re = &data->x_h_re[i];
        data_h_im = &data->x_h_im[i];
        data_v_re = &data->x_v_re[i];
        data_v_im = &data->x_v_im[i];
    }
    else if (port == 2)
    {
        data_h_re = &data->y_h_re[i];
        data_h_im = &data->y_h_im[i];
        data_v_re = &data->y_v_re[i];
        data_v_im = &data->y_v_im[i];
    }

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Read the first line to check units and coordinate system. */
    if (oskar_getline(&line, &bufsize, file) < 0)
    {
        *status = OSKAR_ERR_FILE_IO;
        free(line);
        fclose(file);
        return;
    }

    /* Check for presence of "dBi". */
    dbi = strstr(line, "dBi");

    /* Check for data in Ludwig-3 polarisation system. */
    ludwig3 = strstr(line, "Horiz");

    /* Create local arrays to hold data for fitting. */
    theta  = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    phi    = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    h_re   = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    h_im   = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    v_re   = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    v_im   = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    weight = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    if (*status) return;

    /* Loop over and read each line in the file. */
    n = 0;
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        double t = 0., p = 0., abs_theta_horiz = 0.0, phase_theta_horiz = 0.0;
        double abs_phi_verti = 0.0, phase_phi_verti = 0.0;
        double h_re_ = 0.0, h_im_ = 0.0, v_re_ = 0.0, v_im_ = 0.0;
        void *p_theta = 0, *p_phi = 0, *p_h_re = 0, *p_h_im = 0, *p_v_re = 0;
        void *p_v_im = 0, *p_weight = 0;

        /* Parse the line, and skip if data were not read correctly. */
        /* NOLINTNEXTLINE: Using sscanf() here is clearer than strtod(). */
        if (sscanf(line, "%lf %lf %*f %lf %lf %lf %lf %*f", &t, &p,
                    &abs_theta_horiz, &phase_theta_horiz,
                    &abs_phi_verti, &phase_phi_verti) != 6)
        {
            continue;
        }

        /* Ignore data below horizon if requested. */
        if (ignore_below_horizon && t > 90.0) continue;

        /* Ignore data at poles if requested. */
        if (ignore_at_poles)
        {
            if (t < 1e-6 || t > (180.0 - 1e-6)) continue;
        }

        /* Convert angular measures to radians. */
        t *= DEG2RAD;
        p *= DEG2RAD;
        phase_theta_horiz *= DEG2RAD;
        phase_phi_verti *= DEG2RAD;

        /* Ensure enough space in arrays. */
        if (n % 100 == 0)
        {
            const int size = n + 100;
            oskar_mem_realloc(theta, size, status);
            oskar_mem_realloc(phi, size, status);
            oskar_mem_realloc(h_re, size, status);
            oskar_mem_realloc(h_im, size, status);
            oskar_mem_realloc(v_re, size, status);
            oskar_mem_realloc(v_im, size, status);
            oskar_mem_realloc(weight, size, status);
            if (*status) break;
        }
        p_theta  = oskar_mem_void(theta);
        p_phi    = oskar_mem_void(phi);
        p_h_re   = oskar_mem_void(h_re);
        p_h_im   = oskar_mem_void(h_im);
        p_v_re   = oskar_mem_void(v_re);
        p_v_im   = oskar_mem_void(v_im);
        p_weight = oskar_mem_void(weight);

        /* Convert decibel to linear scale if necessary. */
        if (dbi)
        {
            abs_theta_horiz = pow(10.0, abs_theta_horiz / 10.0);
            abs_phi_verti   = pow(10.0, abs_phi_verti / 10.0);
        }

        /* Amp,phase to real,imag conversion. */
        const double theta_horiz_re = abs_theta_horiz * cos(phase_theta_horiz);
        const double theta_horiz_im = abs_theta_horiz * sin(phase_theta_horiz);
        const double phi_verti_re = abs_phi_verti * cos(phase_phi_verti);
        const double phi_verti_im = abs_phi_verti * sin(phase_phi_verti);

        /* Convert to Ludwig-3 polarisation system if required. */
        if (ludwig3)
        {
            /* Already in Ludwig-3: No conversion required. */
            h_re_ = theta_horiz_re;
            h_im_ = theta_horiz_im;
            v_re_ = phi_verti_re;
            v_im_ = phi_verti_im;
        }
        else
        {
            /* Convert from theta/phi to Ludwig-3. */
            const double sin_p = sin(p);
            const double cos_p = cos(p);
            h_re_ = theta_horiz_re * cos_p - phi_verti_re * sin_p;
            h_im_ = theta_horiz_im * cos_p - phi_verti_im * sin_p;
            v_re_ = theta_horiz_re * sin_p + phi_verti_re * cos_p;
            v_im_ = theta_horiz_im * sin_p + phi_verti_im * cos_p;
        }

        /* Store the surface data in Ludwig-3 format. */
        ((double*)p_theta)[n]  = t;
        ((double*)p_phi)[n]    = p;
        ((double*)p_h_re)[n]   = h_re_;
        ((double*)p_h_im)[n]   = h_im_;
        ((double*)p_v_re)[n]   = v_re_;
        ((double*)p_v_im)[n]   = v_im_;
        ((double*)p_weight)[n] = 1.0;

        /* Increment array pointer. */
        n++;
    }

    /* Free the line buffer and close the file. */
    free(line);
    fclose(file);

    /* Fit splines to the surface data. */
    fit_splines(data_h_re, n, theta, phi, h_re, weight,
            closeness, closeness_inc, "H [real]", log, status);
    fit_splines(data_h_im, n, theta, phi, h_im, weight,
            closeness, closeness_inc, "H [imag]", log, status);
    fit_splines(data_v_re, n, theta, phi, v_re, weight,
            closeness, closeness_inc, "V [real]", log, status);
    fit_splines(data_v_im, n, theta, phi, v_im, weight,
            closeness, closeness_inc, "V [imag]", log, status);

    /* Copy X to Y if both ports are the same. */
    if (port == 0)
    {
        if (!data->y_h_re[i])
        {
            data->y_h_re[i] = oskar_splines_create(
                    OSKAR_DOUBLE, OSKAR_CPU, status);
        }
        if (!data->y_h_im[i])
        {
            data->y_h_im[i] = oskar_splines_create(
                    OSKAR_DOUBLE, OSKAR_CPU, status);
        }
        if (!data->y_v_re[i])
        {
            data->y_v_re[i] = oskar_splines_create(
                    OSKAR_DOUBLE, OSKAR_CPU, status);
        }
        if (!data->y_v_im[i])
        {
            data->y_v_im[i] = oskar_splines_create(
                    OSKAR_DOUBLE, OSKAR_CPU, status);
        }
        oskar_splines_copy(data->y_h_re[i], data->x_h_re[i], status);
        oskar_splines_copy(data->y_h_im[i], data->x_h_im[i], status);
        oskar_splines_copy(data->y_v_re[i], data->x_v_re[i], status);
        oskar_splines_copy(data->y_v_im[i], data->x_v_im[i], status);
    }

    /* Free local arrays. */
    oskar_mem_free(theta, status);
    oskar_mem_free(phi, status);
    oskar_mem_free(h_re, status);
    oskar_mem_free(h_im, status);
    oskar_mem_free(v_re, status);
    oskar_mem_free(v_im, status);
    oskar_mem_free(weight, status);
}


static void fit_splines(oskar_Splines** splines_ptr, int n,
        oskar_Mem* theta, oskar_Mem* phi, oskar_Mem* data, oskar_Mem* weight,
        double closeness, double closeness_inc, const char* name,
        oskar_Log* log, int* status)
{
    double avg_frac_error = 0.0;
    if (*status) return;
    if (!*splines_ptr)
    {
        *splines_ptr = oskar_splines_create(OSKAR_DOUBLE, OSKAR_CPU, status);
    }
    oskar_Splines* splines = *splines_ptr;
    if (*status) return;
    avg_frac_error = closeness; /* Copy the fitting parameter. */
    oskar_log_line(log, 'M', ' ');
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
