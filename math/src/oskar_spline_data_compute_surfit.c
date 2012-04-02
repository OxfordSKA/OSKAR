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

#include "extern/dierckx/surfit.h"
#include "math/oskar_spline_data_compute_surfit.h"
#include "math/oskar_spline_data_init.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_realloc.h"
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>


#define USE_FORTRAN_SURFIT


#ifdef __cplusplus
extern "C" {
#endif

/* Returns the largest absolute (real) value in the array. */
static float max_abs_f(const oskar_Mem* data, int n)
{
    int i;
    float r = -FLT_MAX;
    const float *p;
    p = (const float*)data->data;
    for (i = 0; i < n; ++i)
    {
        if (fabsf(p[i]) > r) r = fabsf(p[i]);
    }
    return r;
}

/* Returns the largest absolute (real) value in the array. */
static double max_abs_d(const oskar_Mem* data, int n)
{
    int i;
    double r = -DBL_MAX;
    const double *p;
    p = (const double*)data->data;
    for (i = 0; i < n; ++i)
    {
        if (fabs(p[i]) > r) r = fabs(p[i]);
    }
    return r;
}

/* Returns the largest value in the array. */
static float max_f(const oskar_Mem* data, int n)
{
    int i;
    float r = -FLT_MAX;
    const float *p;
    p = (const float*)data->data;
    for (i = 0; i < n; ++i)
    {
        if (p[i] > r) r = p[i];
    }
    return r;
}

/* Returns the largest value in the array. */
static double max_d(const oskar_Mem* data, int n)
{
    int i;
    double r = -DBL_MAX;
    const double *p;
    p = (const double*)data->data;
    for (i = 0; i < n; ++i)
    {
        if (p[i] > r) r = p[i];
    }
    return r;
}

/* Returns the smallest value in the array. */
static float min_f(const oskar_Mem* data, int n)
{
    int i;
    float r = FLT_MAX;
    const float *p;
    p = (const float*)data->data;
    for (i = 0; i < n; ++i)
    {
        if (p[i] < r) r = p[i];
    }
    return r;
}

/* Returns the smallest value in the array. */
static double min_d(const oskar_Mem* data, int n)
{
    int i;
    double r = DBL_MAX;
    const double *p;
    p = (const double*)data->data;
    for (i = 0; i < n; ++i)
    {
        if (p[i] < r) r = p[i];
    }
    return r;
}

#ifdef USE_FORTRAN_SURFIT
/* Fortran function prototype. */
void surfit_(int* iopt, int* m, float* x, float* y, const float* z,
        const float* w, float* xb, float* xe, float* yb, float* ye, int* kx,
        int* ky, float* s, int* nxest, int* nyest, int* nmax, float* eps,
        int* nx, float* tx, int* ny, float* ty, float* c, float* fp,
        float* wrk1, int* lwrk1, float* wrk2, int* lwrk2, int* iwrk, int* kwrk,
        int* ier);
#endif

int oskar_spline_data_compute_surfit(oskar_SplineData* spline,
        int num_points, oskar_Mem* x, oskar_Mem* y, const oskar_Mem* data_re,
        const oskar_Mem* data_im, const oskar_Mem* weight_re,
        const oskar_Mem* weight_im, int search, double avg_fractional_err,
        double s_real, double s_imag)
{
    int element_size, err, i, k = 0, maxiter = 1000, type;
    int b1, b2, bx, by, iopt, km, kwrk, lwrk1, lwrk2, ne, nxest, nyest, u, v;
    int sqrt_num_points;
    int *iwrk;
    void *wrk1, *wrk2;
    double factor, factor_fraction;

    /* Order of splines - do not change these values. */
    int kx = 3, ky = 3;

    /* Set values (make these parameters). */
    factor = 0.9;
    factor_fraction = 1.5;

    /* Get the data type. */
    type = data_re->type;
    element_size = oskar_mem_element_size(type);
    if ((type != OSKAR_SINGLE) && (type != OSKAR_DOUBLE))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that input data is on the CPU. */
    if (data_re->location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Initialise and allocate spline data. */
    sqrt_num_points = (int)sqrt(num_points);
    nxest = kx + 1 + sqrt_num_points;
    nyest = ky + 1 + sqrt_num_points;
    u = nxest - kx - 1;
    v = nyest - ky - 1;
    err = oskar_spline_data_init(spline, type, OSKAR_LOCATION_CPU);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_x_re, nxest);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_y_re, nyest);
    if (err) return err;
    err = oskar_mem_realloc(&spline->coeff_re, u * v);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_x_im, nxest);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_y_im, nyest);
    if (err) return err;
    err = oskar_mem_realloc(&spline->coeff_im, u * v);
    if (err) return err;

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
        return OSKAR_ERR_MEMORY_ALLOC_FAILURE;

    if (type == OSKAR_SINGLE)
    {
        /* Set up the surface fitting parameters. */
        float s, user_s, fp = 0.0;
        float eps = 4e-4; /* Important parameter! */
        float x_beg, x_end, y_beg, y_end;
        x_beg = min_f(x, num_points);
        x_end = max_f(x, num_points);
        y_beg = min_f(y, num_points);
        y_end = max_f(y, num_points);

        for (i = 0; i < 2; ++i)
        {
            float *knots_x, *knots_y, *coeff, peak_abs, avg_frac_err_loc;
            const float *data, *weight;
            int *num_knots_x, *num_knots_y, done = 0;
            avg_frac_err_loc = (float)avg_fractional_err;
            if (i == 0) /* Real part. */
            {
                knots_x     = (float*)spline->knots_x_re.data;
                knots_y     = (float*)spline->knots_y_re.data;
                coeff       = (float*)spline->coeff_re.data;
                data        = (const float*)data_re->data;
                weight      = (const float*)weight_re->data;
                num_knots_x = &spline->num_knots_x_re;
                num_knots_y = &spline->num_knots_y_re;
                peak_abs    = max_abs_f(data_re, num_points);
                user_s      = (float)s_real;
            }
            else /* Imaginary part. */
            {
                knots_x     = (float*)spline->knots_x_im.data;
                knots_y     = (float*)spline->knots_y_im.data;
                coeff       = (float*)spline->coeff_im.data;
                data        = (const float*)data_im->data;
                weight      = (const float*)weight_im->data;
                num_knots_x = &spline->num_knots_x_im;
                num_knots_y = &spline->num_knots_y_im;
                peak_abs    = max_abs_f(data_im, num_points);
                user_s      = (float)s_imag;
            }
            do
            {
                float avg_err, term;
                avg_err = avg_frac_err_loc * peak_abs;
                term = num_points * avg_err * avg_err; /* Termination. */
                s = search ? 2.0 * term : user_s; /* Smoothing factor. */
                for (k = 0, iopt = 0; k < maxiter; ++k)
                {
                    if (k > 0) iopt = 1; /* Set iopt to 1 if not first pass. */
#ifdef USE_FORTRAN_SURFIT
                    surfit_(&iopt, &num_points, (float*)x->data,
                            (float*)y->data, data, weight, &x_beg, &x_end,
                            &y_beg, &y_end, &kx, &ky, &s, &nxest, &nyest,
                            &ne, &eps, num_knots_x, knots_x, num_knots_y,
                            knots_y, coeff, &fp, (float*)wrk1, &lwrk1,
                            (float*)wrk2, &lwrk2, iwrk, &kwrk, &err);
#else
                    surfit_f(iopt, num_points, (float*)x->data,
                            (float*)y->data, data, weight, x_beg, x_end,
                            y_beg, y_end, kx, ky, s, nxest, nyest, ne, eps,
                            num_knots_x, knots_x, num_knots_y, knots_y, coeff,
                            &fp, (float*)wrk1, lwrk1, (float*)wrk2, lwrk2,
                            iwrk, kwrk, &err);
#endif
                    printf("Iteration %d, s = %.4e, fp = %.4e\n", k, s, fp);

                    /* Check for errors. */
                    if (err > 0 || err < -2) break;
                    else if (err == -2) s = fp;

                    /* Check if the fit is good enough. */
                    if (!search || fp < term || s < term) break;

                    /* Decrease smoothing factor. */
                    s *= factor;
                }

                /* Check for errors. */
                if (err > 0 || err < -2)
                {
                    printf("Error (%d) finding spline coefficients.\n", err);
                    if (!search || err == 10)
                    {
                        err = OSKAR_ERR_SPLINE_COEFF_FAIL;
                        goto stop;
                    }
                    avg_frac_err_loc *= factor_fraction;
                    printf("Increasing allowed average fractional "
                            "error to %.3f.\n", avg_frac_err_loc);
                }
                else
                {
                    done = 1;
                    err = 0;
                    if (err == 5)
                    {
                        printf("Cannot add any more knots.\n");
                        avg_frac_err_loc = sqrt(fp / num_points) / peak_abs;
                    }
                    if (search)
                    {
                        printf("%s surface fit to %.3f avg. frac. error "
                                "(s=%.2e, fp=%.2e, k=%d).\n",
                                (i == 0 ? "Real" : "Imag"), avg_frac_err_loc,
                                s, fp, k);
                    }
                    else
                    {
                        printf("%s surface fit (s=%.2e, fp=%.2e).\n",
                                (i == 0 ? "Real" : "Imag"), s, fp);
                    }
                    printf("Number of knots (x: %d, y: %d)\n",
                            *num_knots_x, *num_knots_y);
                }
            } while (search && !done);
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        /* Set up the surface fitting parameters. */
        double s, user_s, fp = 0.0;
        double eps = 2e-8; /* Important parameter! */
        double x_beg, x_end, y_beg, y_end;
        x_beg = min_d(x, num_points);
        x_end = max_d(x, num_points);
        y_beg = min_d(y, num_points);
        y_end = max_d(y, num_points);

        for (i = 0; i < 2; ++i)
        {
            double *knots_x, *knots_y, *coeff, peak_abs, avg_frac_err_loc;
            const double *data, *weight;
            int *num_knots_x, *num_knots_y, done = 0;
            avg_frac_err_loc = avg_fractional_err;
            if (i == 0) /* Real part. */
            {
                knots_x     = (double*)spline->knots_x_re.data;
                knots_y     = (double*)spline->knots_y_re.data;
                coeff       = (double*)spline->coeff_re.data;
                data        = (const double*)data_re->data;
                weight      = (const double*)weight_re->data;
                num_knots_x = &spline->num_knots_x_re;
                num_knots_y = &spline->num_knots_y_re;
                peak_abs    = max_abs_d(data_re, num_points);
                user_s      = s_real;
            }
            else /* Imaginary part. */
            {
                knots_x     = (double*)spline->knots_x_im.data;
                knots_y     = (double*)spline->knots_y_im.data;
                coeff       = (double*)spline->coeff_im.data;
                data        = (const double*)data_im->data;
                weight      = (const double*)weight_im->data;
                num_knots_x = &spline->num_knots_x_im;
                num_knots_y = &spline->num_knots_y_im;
                peak_abs    = max_abs_d(data_im, num_points);
                user_s      = s_imag;
            }
            do
            {
                double avg_err, term;
                avg_err = avg_frac_err_loc * peak_abs;
                term = num_points * avg_err * avg_err; /* Termination. */
                s = search ? 2.0 * term : user_s; /* Smoothing factor. */
                for (k = 0, iopt = 0; k < maxiter; ++k)
                {
                    if (k > 0) iopt = 1; /* Set iopt to 1 if not first pass. */
                    surfit_d(iopt, num_points, (double*)x->data,
                            (double*)y->data, data, weight, x_beg, x_end,
                            y_beg, y_end, kx, ky, s, nxest, nyest, ne, eps,
                            num_knots_x, knots_x, num_knots_y, knots_y, coeff,
                            &fp, (double*)wrk1, lwrk1, (double*)wrk2, lwrk2,
                            iwrk, kwrk, &err);
                    printf("Iteration %d, s = %.4e, fp = %.4e\n", k, s, fp);

                    /* Check for errors. */
                    if (err > 0 || err < -2) break;
                    else if (err == -2) s = fp;

                    /* Check if the fit is good enough. */
                    if (!search || fp < term || s < term) break;

                    /* Decrease smoothing factor. */
                    s *= factor;
                }

                /* Check for errors. */
                if (err > 0 || err < -2)
                {
                    printf("Error (%d) finding spline coefficients.\n", err);
                    if (!search || err == 10)
                    {
                        err = OSKAR_ERR_SPLINE_COEFF_FAIL;
                        goto stop;
                    }
                    avg_frac_err_loc *= factor_fraction;
                    printf("Increasing allowed average fractional "
                            "error to %.3f.\n", avg_frac_err_loc);
                }
                else
                {
                    done = 1;
                    err = 0;
                    if (err == 5)
                    {
                        printf("Cannot add any more knots.\n");
                        avg_frac_err_loc = sqrt(fp / num_points) / peak_abs;
                    }
                    if (search)
                    {
                        printf("%s surface fit to %.3f avg. frac. error "
                                "(s=%.2e, fp=%.2e, k=%d).\n",
                                (i == 0 ? "Real" : "Imag"), avg_frac_err_loc,
                                s, fp, k);
                    }
                    else
                    {
                        printf("%s surface fit (s=%.2e, fp=%.2e).\n",
                                (i == 0 ? "Real" : "Imag"), s, fp);
                    }
                    printf("Number of knots (x: %d, y: %d)\n",
                            *num_knots_x, *num_knots_y);
                }
            } while (search && !done);
        }
    }

    /* Free work arrays. */
stop:
    free(iwrk);
    free(wrk2);
    free(wrk1);

    return err;
}

#ifdef __cplusplus
}
#endif
