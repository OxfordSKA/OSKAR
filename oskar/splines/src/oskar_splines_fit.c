/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "splines/oskar_dierckx_sphere.h"
#include "splines/oskar_dierckx_surfit.h"
#include "splines/private_splines.h"
#include "splines/oskar_splines.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns the range of values in the array. */
static void min_max(const double* data, int n, double* min, double* max)
{
    int i = 0;
    double val = 0.0;
    *max = -DBL_MAX, *min = DBL_MAX;
    for (i = 0; i < n; ++i)
    {
        val = data[i];
        if (val > *max) *max = val;
        if (val < *min) *min = val;
    }
}

void oskar_splines_fit(oskar_Splines* spline, int num_points, double* x_theta,
        double* y_phi, const double* z_data, const double* weight,
        int fit_type, int search_flag, double* avg_frac_err,
        double inc_factor, double smooth_factor, double epsilon, int* status)
{
    int err = 0, kwrk = 0, lwrk1 = 0, lwrk2 = 0, u = 0, v = 0, *iwrk = 0;
    double *coeff = 0, *wrk1 = 0, *wrk2 = 0;
    double avg_err = 0., peak_abs = 0., fp = 0., z_min = 0., z_max = 0.;
    if (*status) return;
    if (num_points <= 0) return;

    /* Check the output data type and location. */
    if (oskar_splines_precision(spline) != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (oskar_splines_mem_location(spline) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check that parameters are within allowed ranges. */
    if (inc_factor <= 1.0)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Get range of z_data. */
    min_max(z_data, num_points, &z_min, &z_max);
    peak_abs = fabs(z_min) > fabs(z_max) ? fabs(z_min) : fabs(z_max);

    /* Check range. */
    if ((z_max - z_min) < epsilon)
    {
        /* No surface to fit, so return. */
        return;
    }

    /* Check fit type. */
    if (fit_type == OSKAR_SPLINES_LINEAR)
    {
        int b1 = 0, b2 = 0, bx = 0, by = 0, km = 0;
        int nxest = 0, nyest = 0, ne = 0;
        double x_min = 0.0, x_max = 0.0, y_min = 0.0, y_max = 0.0;
        double *knots_x = 0, *knots_y = 0;

        /* Order of splines - do not change these values. */
        int kx = 3, ky = 3;

        /* Get data boundaries. */
        min_max(x_theta, num_points, &x_min, &x_max);
        min_max(y_phi, num_points, &y_min, &y_max);

        /* Allocate output spline data arrays. */
        nxest = kx + 1 + (int)ceil(1.5 * sqrt(num_points));
        nyest = ky + 1 + (int)ceil(1.5 * sqrt(num_points));
        u = nxest - kx - 1;
        v = nyest - ky - 1;
        oskar_mem_realloc(spline->knots_x_theta, nxest, status);
        oskar_mem_realloc(spline->knots_y_phi, nyest, status);
        oskar_mem_realloc(spline->coeff, u * v, status);

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
        lwrk1 = 1 + b2 + u * v * (2 + b1 + b2) +
                2 * (u + v + km * (num_points + ne) + ne - kx - ky);
        lwrk2 = u * v * (b2 + 1) + b2;
        kwrk = num_points + (nxest - 2 * kx - 1) * (nyest - 2 * ky - 1);
        wrk1 = (double*)malloc(lwrk1 * sizeof(double));
        wrk2 = (double*)malloc(lwrk2 * sizeof(double));
        iwrk = (int*)malloc(kwrk * sizeof(int));
        if (!wrk1 || !wrk2 || !iwrk)
        {
            free(wrk1);
            free(wrk2);
            free(iwrk);
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return;
        }

        /* Fitting procedure. */
        knots_x = oskar_mem_double(spline->knots_x_theta, status);
        knots_y = oskar_mem_double(spline->knots_y_phi, status);
        coeff   = oskar_mem_double(spline->coeff, status);
        do
        {
            avg_err = *avg_frac_err * peak_abs;
            spline->smoothing_factor = search_flag ?
                    (num_points * avg_err * avg_err) : smooth_factor;
            oskar_dierckx_surfit(0, num_points, x_theta, y_phi, z_data,
                    weight, x_min, x_max, y_min, y_max, kx, ky,
                    spline->smoothing_factor, nxest, nyest, ne, epsilon,
                    &spline->num_knots_x_theta, knots_x,
                    &spline->num_knots_y_phi, knots_y, coeff, &fp, wrk1,
                    lwrk1, wrk2, lwrk2, iwrk, kwrk, &err);

            /* Break immediately if successful. */
            if (err == 0 || err == -1 || err == -2) break;

            /* Check for unrecoverable errors. */
            if (!search_flag || err >= 10 || *avg_frac_err == 0.0)
            {
                *status = OSKAR_ERR_SPLINE_COEFF_FAIL;
                break;
            }
            else
            {
                err = 0; /* Try again with a larger smoothing factor. */
                *avg_frac_err *= inc_factor;
            }
        } while (search_flag);

        /* Compact the knot and coefficient arrays. */
        u = spline->num_knots_x_theta - kx - 1;
        v = spline->num_knots_y_phi - ky - 1;
        oskar_mem_realloc(spline->knots_x_theta, spline->num_knots_x_theta,
                status);
        oskar_mem_realloc(spline->knots_y_phi, spline->num_knots_y_phi, status);
        oskar_mem_realloc(spline->coeff, u * v, status);
    }
    else if (fit_type == OSKAR_SPLINES_SPHERICAL)
    {
        int ntest = 0, npest = 0;
        double *knots_theta = 0, *knots_phi = 0;

        /* Allocate output spline data arrays. */
        ntest = 8 + (int)ceil(sqrt(num_points/2));
        npest = 8 + (int)ceil(sqrt(num_points/2));
        oskar_mem_realloc(spline->knots_x_theta, ntest, status);
        oskar_mem_realloc(spline->knots_y_phi, npest, status);
        oskar_mem_realloc(spline->coeff, (ntest-4) * (npest-4), status);

        /* Check if safe to proceed. */
        if (*status) return;

        /* Set up workspace. */
        u = ntest - 7;
        v = npest - 7;
        lwrk1 = 185 + 52*v + 10*u + 14*u*v + 8*(u-1)*v*v + 8*num_points;
        lwrk2 = 48 + 21*v + 7*u*v + 4*(u-1)*v*v;
        kwrk = num_points + u*v;
        wrk1 = (double*)malloc(lwrk1 * sizeof(double));
        wrk2 = (double*)malloc(lwrk2 * sizeof(double));
        iwrk = (int*)malloc(kwrk * sizeof(int));
        if (!wrk1 || !wrk2 || !iwrk)
        {
            free(wrk1);
            free(wrk2);
            free(iwrk);
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return;
        }

        /* Fitting procedure. */
        knots_theta = oskar_mem_double(spline->knots_x_theta, status);
        knots_phi   = oskar_mem_double(spline->knots_y_phi, status);
        coeff       = oskar_mem_double(spline->coeff, status);
        do
        {
            avg_err = *avg_frac_err * peak_abs;
            spline->smoothing_factor = search_flag ?
                    (num_points * avg_err * avg_err) : smooth_factor;
            oskar_dierckx_sphere(0, num_points, x_theta, y_phi, z_data, weight,
                    spline->smoothing_factor, ntest, npest, epsilon,
                    &spline->num_knots_x_theta, knots_theta,
                    &spline->num_knots_y_phi, knots_phi, coeff, &fp, wrk1,
                    lwrk1, wrk2, lwrk2, iwrk, kwrk, &err);

            /* Break immediately if successful. */
            if (err == 0 || err == -1 || err == -2) break;

            /* Check for unrecoverable errors. */
            if (!search_flag || err >= 10 || *avg_frac_err == 0.0)
            {
                *status = OSKAR_ERR_SPLINE_COEFF_FAIL;
                break;
            }
            else
            {
                err = 0; /* Try again with a larger smoothing factor. */
                *avg_frac_err *= inc_factor;
            }
        } while (search_flag);

        /* Compact the knot and coefficient arrays. */
        u = spline->num_knots_x_theta - 4;
        v = spline->num_knots_y_phi - 4;
        oskar_mem_realloc(spline->knots_x_theta, spline->num_knots_x_theta,
                status);
        oskar_mem_realloc(spline->knots_y_phi, spline->num_knots_y_phi, status);
        oskar_mem_realloc(spline->coeff, u * v, status);
    }

    /* Free work arrays. */
    free(wrk1);
    free(wrk2);
    free(iwrk);
}

#ifdef __cplusplus
}
#endif
