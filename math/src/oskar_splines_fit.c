/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <oskar_dierckx_surfit.h>
#include <private_splines.h>
#include <oskar_splines.h>
#include <oskar_log.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns the largest absolute (real) value in the array. */
static double oskar_mem_max_abs(const oskar_Mem* data, int n)
{
    int i;
    double r = -DBL_MAX;
    if (oskar_mem_type(data) == OSKAR_SINGLE)
    {
        const float *p;
        p = (const float*)oskar_mem_void_const(data);
        for (i = 0; i < n; ++i)
        {
            if (fabsf(p[i]) > r) r = fabsf(p[i]);
        }
    }
    else if (oskar_mem_type(data) == OSKAR_DOUBLE)
    {
        const double *p;
        p = (const double*)oskar_mem_void_const(data);
        for (i = 0; i < n; ++i)
        {
            if (fabs(p[i]) > r) r = fabs(p[i]);
        }
    }
    return r;
}

/* Returns the largest value in the array. */
static double oskar_mem_max(const oskar_Mem* data, int n)
{
    int i;
    double r = -DBL_MAX;
    if (oskar_mem_type(data) == OSKAR_SINGLE)
    {
        const float *p;
        p = (const float*)oskar_mem_void_const(data);
        for (i = 0; i < n; ++i)
        {
            if (p[i] > r) r = p[i];
        }
    }
    else if (oskar_mem_type(data) == OSKAR_DOUBLE)
    {
        const double *p;
        p = (const double*)oskar_mem_void_const(data);
        for (i = 0; i < n; ++i)
        {
            if (p[i] > r) r = p[i];
        }
    }
    return r;
}

/* Returns the smallest value in the array. */
static double oskar_mem_min(const oskar_Mem* data, int n)
{
    int i;
    double r = DBL_MAX;
    if (oskar_mem_type(data) == OSKAR_SINGLE)
    {
        const float *p;
        p = (const float*)oskar_mem_void_const(data);
        for (i = 0; i < n; ++i)
        {
            if (p[i] < r) r = p[i];
        }
    }
    else if (oskar_mem_type(data) == OSKAR_DOUBLE)
    {
        const double *p;
        p = (const double*)oskar_mem_void_const(data);
        for (i = 0; i < n; ++i)
        {
            if (p[i] < r) r = p[i];
        }
    }
    return r;
}

void oskar_splines_fit(oskar_Splines* spline, oskar_Log* log,
        int num_points, oskar_Mem* x, oskar_Mem* y, const oskar_Mem* z,
        const oskar_Mem* w, const oskar_SettingsSpline* settings, int* status)
{
    int element_size, err = 0, type;
    int b1, b2, bx, by, km, kwrk, lwrk1, lwrk2, ne, nxest, nyest, u, v, *iwrk;
    int search_flag, sqrt_num_points, done = 0;
    void *wrk1, *wrk2;
    double x_beg, x_end, y_beg, y_end, eps;
    double avg_err, avg_frac_err, peak_abs, s, user_s, factor;

    /* Order of splines - do not change these values. */
    int kx = 3, ky = 3;

    /* Check all inputs. */
    if (!spline || !x || !y || !z || !w || !settings || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;
    if (num_points <= 0) return;

    /* Get the data type. */
    type = oskar_mem_type(z);
    element_size = oskar_mem_element_size(type);
    if ((type != OSKAR_SINGLE) && (type != OSKAR_DOUBLE))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Check that parameters are within allowed ranges. */
    eps = (type == OSKAR_SINGLE) ? settings->eps_float : settings->eps_double;
    search_flag  = settings->search_for_best_fit;
    avg_frac_err = settings->average_fractional_error;
    user_s       = settings->smoothness_factor_override;
    factor       = settings->average_fractional_error_factor_increase;
    if (factor <= 1.0)
    {
        *status = OSKAR_ERR_SETTINGS;
        return;
    }

    /* Check that input data is on the CPU. */
    if (oskar_mem_location(x) != OSKAR_LOCATION_CPU ||
            oskar_mem_location(y) != OSKAR_LOCATION_CPU ||
            oskar_mem_location(z) != OSKAR_LOCATION_CPU ||
            oskar_mem_location(w) != OSKAR_LOCATION_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Get data boundaries. */
    x_beg    = oskar_mem_min(x, num_points);
    x_end    = oskar_mem_max(x, num_points);
    y_beg    = oskar_mem_min(y, num_points);
    y_end    = oskar_mem_max(y, num_points);
    peak_abs = oskar_mem_max_abs(z, num_points);

    /* Initialise and allocate spline data. */
    sqrt_num_points = (int)sqrt(num_points);
    nxest = kx + 1 + 3 * sqrt_num_points / 2;
    nyest = ky + 1 + 3 * sqrt_num_points / 2;
    u = nxest - kx - 1;
    v = nyest - ky - 1;
    oskar_mem_realloc(&spline->knots_x, nxest, status);
    oskar_mem_realloc(&spline->knots_y, nyest, status);
    oskar_mem_realloc(&spline->coeff, u * v, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Set up workspace. */
    km = 1 + ((kx > ky) ? kx : ky);
    ne = (nxest > nyest) ? nxest : nyest;
    bx = kx * v + ky + 1;
    by = ky * u + kx + 1;
    if (bx <= by)
    {
        b1 = bx;
        b2 = b1 + v - ky;
    }
    else
    {
        b1 = by;
        b2 = b1 + u - kx;
    }
    lwrk1 = u * v * (2 + b1 + b2) +
            2 * (u + v + km * (num_points + ne) + ne - kx - ky) + b2 + 1;
    lwrk2 = u * v * (b2 + 1) + b2;
    kwrk = num_points + (nxest - 2 * kx - 1) * (nyest - 2 * ky - 1);
    wrk1 = malloc(lwrk1 * element_size);
    wrk2 = malloc(lwrk2 * element_size);
    iwrk = (int*)malloc(kwrk * sizeof(int));
    if (wrk1 == NULL || wrk2 == NULL || iwrk == NULL)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return;
    }

    if (type == OSKAR_SINGLE)
    {
        float fp = 0.0, *knots_x, *knots_y, *coeff, *x_, *y_;
        const float *z_, *w_;
        knots_x = oskar_mem_float(&spline->knots_x, status);
        knots_y = oskar_mem_float(&spline->knots_y, status);
        coeff   = oskar_mem_float(&spline->coeff, status);
        x_      = oskar_mem_float(x, status);
        y_      = oskar_mem_float(y, status);
        z_      = oskar_mem_float_const(z, status);
        w_      = oskar_mem_float_const(w, status);
        do
        {
            avg_err = avg_frac_err * peak_abs;
            s = search_flag ? (num_points * avg_err * avg_err) : user_s;
            oskar_dierckx_surfit_f(0, num_points, x_, y_, z_, w_,
                    (float)x_beg, (float)x_end, (float)y_beg, (float)y_end,
                    kx, ky, (float)s, nxest, nyest, ne, (float)eps,
                    &spline->num_knots_x, knots_x, &spline->num_knots_y,
                    knots_y, coeff, &fp, (float*)wrk1, lwrk1, (float*)wrk2,
                    lwrk2, iwrk, kwrk, &err);

            /* Check for errors. */
            if (err == 0 || err == -1 || err == -2) done = 1;
            else
            {
                if (!search_flag || err >= 10 || avg_frac_err == 0.0)
                {
                    *status = OSKAR_ERR_SPLINE_COEFF_FAIL;
                    done = 1;
                }
                else
                {
                    err = 0; /* Try again with a larger smoothing factor. */
                    avg_frac_err *= factor;
                }
            }
        } while (search_flag && !done);
    }
    else if (type == OSKAR_DOUBLE)
    {
        double fp = 0.0, *knots_x, *knots_y, *coeff, *x_, *y_;
        const double *z_, *w_;
        knots_x = oskar_mem_double(&spline->knots_x, status);
        knots_y = oskar_mem_double(&spline->knots_y, status);
        coeff   = oskar_mem_double(&spline->coeff, status);
        x_      = oskar_mem_double(x, status);
        y_      = oskar_mem_double(y, status);
        z_      = oskar_mem_double_const(z, status);
        w_      = oskar_mem_double_const(w, status);
        do
        {
            avg_err = avg_frac_err * peak_abs;
            s = search_flag ? (num_points * avg_err * avg_err) : user_s;
            oskar_dierckx_surfit_d(0, num_points, x_, y_, z_, w_,
                    x_beg, x_end, y_beg, y_end, kx, ky, s, nxest, nyest, ne,
                    eps, &spline->num_knots_x, knots_x, &spline->num_knots_y,
                    knots_y, coeff, &fp, (double*)wrk1, lwrk1, (double*)wrk2,
                    lwrk2, iwrk, kwrk, &err);

            /* Check for errors. */
            if (err == 0 || err == -1 || err == -2) done = 1;
            else
            {
                if (!search_flag || err >= 10 || avg_frac_err == 0.0)
                {
                    *status = OSKAR_ERR_SPLINE_COEFF_FAIL;
                    done = 1;
                }
                else
                {
                    err = 0; /* Try again with a larger smoothing factor. */
                    avg_frac_err *= factor;
                }
            }
        } while (search_flag && !done);
    }

    if (!*status)
    {
        if (search_flag)
        {
            oskar_log_message(log, 1, "Surface fitted to %.3f average "
                    "frac. error (s=%.2e).", avg_frac_err, s);
        }
        else
        {
            oskar_log_message(log, 1, "Surface fitted (s=%.2e).", s);
        }
        oskar_log_message(log, 1, "Number of knots (x, y) = (%d, %d).",
                spline->num_knots_x, spline->num_knots_y);
        oskar_log_message(log, 0, "");
    }

    /* Free work arrays. */
    free(iwrk);
    free(wrk2);
    free(wrk1);
}

#ifdef __cplusplus
}
#endif
