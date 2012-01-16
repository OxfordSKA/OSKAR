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

#include "extern/dierckx/sphere.h"
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

/* Returns the largest absolute (real) value in the array. */
static float max_f(const oskar_Mem* data, int n)
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
static double max_d(const oskar_Mem* data, int n)
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

#ifdef __cplusplus
extern "C" {
#endif

int oskar_spline_data_compute_surfit(oskar_SplineData* spline,
        int num_points, const oskar_Mem* theta, const oskar_Mem* phi,
        const oskar_Mem* data_re, const oskar_Mem* data_im,
        const oskar_Mem* weight_re, const oskar_Mem* weight_im, int search,
        double avg_fractional_err, double s_real, double s_imag)
{
    int element_size, err, est, i, iopt, k = 0;
    int kwrk, lwrk1, lwrk2, maxiter = 1000, type, u;
    int *iwrk;
    void *wrk1, *wrk2;

    /* Get the data type. */
    type = data_re->private_type;
    element_size = oskar_mem_element_size(type);
    if ((type != OSKAR_SINGLE) && (type != OSKAR_DOUBLE))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that input data is on the CPU. */
    if (data_re->private_location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Initialise and allocate spline data. */
    est = 1.3 * (8 + (int)sqrt(num_points));
    err = oskar_spline_data_init(spline, type, OSKAR_LOCATION_CPU);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_x_re, est);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_y_re, est);
    if (err) return err;
    err = oskar_mem_realloc(&spline->coeff_re, (est-4)*(est-4));
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_x_im, est);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_y_im, est);
    if (err) return err;
    err = oskar_mem_realloc(&spline->coeff_im, (est-4)*(est-4));
    if (err) return err;

    /* Set up workspace. */
    u = est - 7;
    lwrk1 = 185 + 52*u + 10*u + 14*u*u + 8*(u-1)*u*u + 8*num_points;
    lwrk2 = 48 + 21*u + 7*u*u + 4*(u-1)*u*u;
    kwrk = num_points + u*u;
    wrk1 = malloc(lwrk1 * element_size);
    wrk2 = malloc(lwrk2 * element_size);
    iwrk = (int*)malloc(kwrk * sizeof(int));

    if (type == OSKAR_SINGLE)
    {
        /* Set up the surface fitting parameters. */
        float s, user_s, fp = 0.0;
        float eps = 1e-6; /* Magnitude of float epsilon. */

        for (i = 0; i < 2; ++i)
        {
            float *knots_theta, *knots_phi, *coeff, peak_abs, avg_frac_err_loc;
            const float *data, *weight;
            int *num_knots_theta, *num_knots_phi, done = 0;
            avg_frac_err_loc = (float)avg_fractional_err;
            if (i == 0) /* Real part. */
            {
                knots_theta     = (float*)spline->knots_x_re.data;
                knots_phi       = (float*)spline->knots_y_re.data;
                coeff           = (float*)spline->coeff_re.data;
                data            = (const float*)data_re->data;
                weight          = (const float*)weight_re->data;
                num_knots_theta = &spline->num_knots_x_re;
                num_knots_phi   = &spline->num_knots_y_re;
                peak_abs        = max_f(data_re, num_points);
                user_s          = (float)s_real;
            }
            else /* Imaginary part. */
            {
                knots_theta     = (float*)spline->knots_x_im.data;
                knots_phi       = (float*)spline->knots_y_im.data;
                coeff           = (float*)spline->coeff_im.data;
                data            = (const float*)data_im->data;
                weight          = (const float*)weight_im->data;
                num_knots_theta = &spline->num_knots_x_im;
                num_knots_phi   = &spline->num_knots_y_im;
                peak_abs        = max_f(data_im, num_points);
                user_s          = (float)s_imag;
            }
            do
            {
                float avg_err, term;
                avg_err = avg_frac_err_loc * peak_abs;
                term = num_points * avg_err * avg_err; /* Termination. */
                s = search ? 2.0 * term : user_s; /* Smoothing factor. */
                for (k = 0, iopt = 0; k < maxiter; ++k)
                {
                    if (k > 0) iopt = 1; /* Set iopt to 1 if not the first pass. */
                    /* surfit_(); */
                    printf("Iteration %d, s = %.4e, fp = %.4e\n", k, s, fp);

                    /* Check for errors. */
                    if (err > 0 || err < -2) break;
                    else if (err == -2) s = fp;

                    /* Check if the fit is good enough. */
                    if (!search || fp < term || s < term) break;

                    /* Decrease smoothing factor. */
                    s *= 0.9;
                }

                /* Check for errors. */
                if (err == 5)
                {
                    done = 1;
                    err = 0;
                    printf("Cannot add any more knots.\n");
                    avg_frac_err_loc = sqrt(fp / num_points) / peak_abs;
                    if (search)
                        printf("%s surface fit to %.3f avg. frac. error "
                                "(s=%.2e, fp=%.2e, k=%d).\n",
                                (i == 0 ? "Real" : "Imag"), avg_frac_err_loc,
                                s, fp, k);
                    else
                        printf("%s surface fit (s=%.2e, fp=%.2e).\n",
                                (i == 0 ? "Real" : "Imag"), s, fp);
                    printf("Number of knots (theta: %d, phi: %d)\n",
                            *num_knots_theta, *num_knots_phi);
                }
                else if (err > 0 || err < -2)
                {
                    printf("Error finding spline coefficients (code %d).\n", err);
                    if (!search)
                    {
                        err = OSKAR_ERR_SPLINE_COEFF_FAIL;
                        goto stop;
                    }
                    avg_frac_err_loc *= 2.0;
                    printf("Increasing allowed average fractional "
                            "error to %.3f.\n", avg_frac_err_loc);
                }
                else
                {
                    done = 1;
                    err = 0;
                    if (search)
                        printf("%s surface fit to %.3f avg. frac. error "
                                "(s=%.2e, fp=%.2e, k=%d).\n",
                                (i == 0 ? "Real" : "Imag"), avg_frac_err_loc,
                                s, fp, k);
                    else
                        printf("%s surface fit (s=%.2e, fp=%.2e).\n",
                                (i == 0 ? "Real" : "Imag"), s, fp);
                    printf("Number of knots (theta: %d, phi: %d)\n",
                            *num_knots_theta, *num_knots_phi);
                }
            } while (search && !done);
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
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
