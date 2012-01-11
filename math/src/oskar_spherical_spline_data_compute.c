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
#include "math/oskar_spherical_spline_data_compute.h"
#include "math/oskar_spherical_spline_data_init.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_realloc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void sphere_(int* iopt, int* m, const float* theta, const float* phi,
        const float* r, const float* w, float* s, int* ntest, int* npest,
        float* eps, int* nt, float* tt, int* np, float* tp, float* c,
        float* fp, float* wrk1, int* lwrk1, float* wrk2, int* lwrk2,
        int* iwrk, int* kwrk, int* ier);

int oskar_spherical_spline_data_compute(oskar_SphericalSplineData* spline,
        int num_points, const oskar_Mem* theta, const oskar_Mem* phi,
        const oskar_Mem* data_re, const oskar_Mem* data_im,
        const oskar_Mem* weight)
{
    int element_size, err, est, i, iopt, k = 0;
    int kwrk, lwrk1, lwrk2, maxiter, type, u;
    int *iwrk;
    void *wrk1, *wrk2;
    double noise, s;

    /* Get the data type. */
    type = data_re->private_type;
    element_size = oskar_mem_element_size(type);
    if ((type != OSKAR_SINGLE) && (type != OSKAR_DOUBLE))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that input data is on the CPU. */
    if (data_re->private_location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Initialise and allocate spline data. */
    est = 8 + (int)sqrt(num_points); /* Could use num_points/2 if needed. */
    err = oskar_spherical_spline_data_init(spline, type, OSKAR_LOCATION_CPU);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_theta_re, est);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_phi_re, est);
    if (err) return err;
    err = oskar_mem_realloc(&spline->coeff_re, (est-4)*(est-4));
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_theta_im, est);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_phi_im, est);
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
    noise = 5e-4; /* Numerical noise on input data. */
    maxiter = 1000;

    if (type == OSKAR_SINGLE)
    {
        /* Set up the surface fitting parameters. */
        float fp = 0.0;
        float eps = 1e-5; /* Magnitude of float epsilon. */

        for (i = 0; i < 2; ++i)
        {
            float *knots_theta, *knots_phi, *coeff;
            const float *data;
            int *num_knots_theta, *num_knots_phi;
            if (i == 0) /* Real part. */
            {
                knots_theta     = (float*)spline->knots_theta_re.data;
                knots_phi       = (float*)spline->knots_phi_re.data;
                coeff           = (float*)spline->coeff_re.data;
                data            = (const float*)data_re->data;
                num_knots_theta = &spline->num_knots_theta_re;
                num_knots_phi   = &spline->num_knots_phi_re;
            }
            else /* Imaginary part. */
            {
                knots_theta     = (float*)spline->knots_theta_im.data;
                knots_phi       = (float*)spline->knots_phi_im.data;
                coeff           = (float*)spline->coeff_im.data;
                data            = (const float*)data_im->data;
                num_knots_theta = &spline->num_knots_theta_im;
                num_knots_phi   = &spline->num_knots_phi_im;
            }
            s = (num_points + sqrt(2.0 * num_points)); /* Smoothing factor. */
            for (k = 0, iopt = 0; k < maxiter; ++k)
            {
                float s2;
                s2 = (float)s;
                if (k > 0) iopt = 1; /* Set iopt to 1 if not the first pass. */
                /*
                sphere_f(iopt, num_points, (const float*)theta->data,
                        (const float*)phi->data, data,
                        (const float*)weight->data, (float)s, est, est, eps,
                        num_knots_theta, knots_theta, num_knots_phi, knots_phi,
                        coeff, &fp, (float*)wrk1, lwrk1, (float*)wrk2, lwrk2,
                        iwrk, kwrk, &err);
                 */
                sphere_(&iopt, &num_points, (const float*)theta->data,
                        (const float*)phi->data, data,
                        (const float*)weight->data, &s2, &est, &est, &eps,
                        num_knots_theta, knots_theta, num_knots_phi, knots_phi,
                        coeff, &fp, (float*)wrk1, &lwrk1, (float*)wrk2, &lwrk2,
                        iwrk, &kwrk, &err);

                /* Check return code. */
                if (err > 0 || err < -2)
                {
                    printf("Error computing spline coefficients: code %d.\n", err);
                    err = OSKAR_ERR_SPLINE_COEFF_FAIL;
                    break;
                }
                else if (err == -2) s = fp * 0.9;
                else s /= 1.2;

                /* Check if the fit is good enough. */
                if ((fp / num_points) < pow(2.0 * noise, 2.0)) break;
            }

            /* Check if iteration limit was reached. */
            printf("Surface fit complete ");
            printf("(i = %d, s = %.5e, fp = %.5e, k = %d).\n", i, s, fp, k);
            if (k >= maxiter-1) err = OSKAR_ERR_SPLINE_COEFF_FAIL;
            if (err > 0 || err < -2) break;
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        /* Set up the surface fitting parameters. */
        double fp = 0.0;
        double eps = 1e-16; /* Magnitude of double epsilon. */

        for (i = 0; i < 2; ++i)
        {
            double *knots_theta, *knots_phi, *coeff;
            const double *data;
            int *num_knots_theta, *num_knots_phi;
            if (i == 0) /* Real part. */
            {
                knots_theta     = (double*)spline->knots_theta_re.data;
                knots_phi       = (double*)spline->knots_phi_re.data;
                coeff           = (double*)spline->coeff_re.data;
                data            = (const double*)data_re->data;
                num_knots_theta = &spline->num_knots_theta_re;
                num_knots_phi   = &spline->num_knots_phi_re;
            }
            else /* Imaginary part. */
            {
                knots_theta     = (double*)spline->knots_theta_im.data;
                knots_phi       = (double*)spline->knots_phi_im.data;
                coeff           = (double*)spline->coeff_im.data;
                data            = (const double*)data_im->data;
                num_knots_theta = &spline->num_knots_theta_im;
                num_knots_phi   = &spline->num_knots_phi_im;
            }
            s = num_points + sqrt(2.0 * num_points); /* Smoothing factor. */
            for (k = 0, iopt = 0; k < maxiter; ++k)
            {
                if (k > 0) iopt = 1; /* Set iopt to 1 if not the first pass. */
                sphere_d(iopt, num_points, (const double*)theta->data,
                        (const double*)phi->data, data,
                        (const double*)weight->data, (double)s, est, est, eps,
                        num_knots_theta, knots_theta, num_knots_phi, knots_phi,
                        coeff, &fp, (double*)wrk1, lwrk1, (double*)wrk2, lwrk2,
                        iwrk, kwrk, &err);

                /* Check return code. */
                if (err > 0 || err < -2)
                {
                    printf("Error computing spline coefficients: code %d.\n", err);
                    err = OSKAR_ERR_SPLINE_COEFF_FAIL;
                    break;
                }
                else if (err == -2) s = fp * 0.9;
                else s /= 1.2;

                /* Check if the fit is good enough. */
                if ((fp / num_points) < pow(2.0 * noise, 2.0)) break;
            }

            /* Check if iteration limit was reached. */
            printf("Surface fit complete ");
            printf("(i = %d, s = %.5e, fp = %.5e, k = %d).\n", i, s, fp, k);
            if (k >= maxiter-1) err = OSKAR_ERR_SPLINE_COEFF_FAIL;
            if (err > 0 || err < -2) break;
        }
    }

    /* Free work arrays. */
    free(iwrk);
    free(wrk2);
    free(wrk1);

    return err;
}

#ifdef __cplusplus
}
#endif
