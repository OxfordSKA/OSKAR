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

#include "extern/dierckx/bispev.h"
#include "math/oskar_spline_data_evaluate.h"
#include "utility/oskar_mem_type_check.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fortran function prototype. */
void bispev_(const float tx[], const int* nx, const float ty[], const int* ny,
        const float c[], const int* kx, const int* ky, const float x[],
        const int* mx, const float y[], const int* my, float z[],
        float wrk[], const int* lwrk, int iwrk[], const int* kwrk, int* ier);

static int c1 = 1;
static int c3 = 3;

int oskar_spline_data_evaluate(oskar_Mem* output, int stride,
        const oskar_SplineData* spline, const oskar_Mem* theta,
        const oskar_Mem* phi)
{
    int err = 0, num_points, type, location;

    /* Check arrays are consistent. */
    num_points = theta->private_num_elements;

    /* Check type. */
    type = theta->private_type;
    if (type != phi->private_type)
        return OSKAR_ERR_TYPE_MISMATCH;
    if (!oskar_mem_is_complex(output->private_type))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check location. */
    location = output->private_location;
    if (location != spline->coeff_re.private_location ||
            location != spline->knots_x_re.private_location ||
            location != spline->knots_y_re.private_location ||
            location != theta->private_location ||
            location != phi->private_location)
        return OSKAR_ERR_BAD_LOCATION;

    /* Check if data are in CPU memory. */
    if (location == OSKAR_LOCATION_CPU)
    {
        /* Set up workspace. */
        int i, iwrk1[2], j, kwrk1 = 2, lwrk = 8, nt, np;

        if (type == OSKAR_SINGLE)
        {
            float wrk[8];
            for (j = 0; j < 2; ++j)
            {
                const float *knots_theta, *knots_phi, *coeff;
                float *out;
                if (j == 0) /* Real part. */
                {
                    nt          = spline->num_knots_x_re;
                    np          = spline->num_knots_y_re;
                    knots_theta = (const float*)spline->knots_x_re.data;
                    knots_phi   = (const float*)spline->knots_y_re.data;
                    coeff       = (const float*)spline->coeff_re.data;
                    out         = (float*)output->data;
                }
                else  /* Imaginary part. */
                {
                    nt          = spline->num_knots_x_im;
                    np          = spline->num_knots_y_im;
                    knots_theta = (const float*)spline->knots_x_im.data;
                    knots_phi   = (const float*)spline->knots_y_im.data;
                    coeff       = (const float*)spline->coeff_im.data;
                    out         = (float*)output->data + 1;
                }

                for (i = 0; i < num_points; ++i)
                {
                    float theta1, phi1;
                    theta1 = ((const float*)theta->data)[i];
                    phi1 = ((const float*)phi->data)[i];
                    bispev_(knots_theta, &nt, knots_phi, &np, coeff, &c3, &c3,
                            &theta1, &c1, &phi1, &c1, &out[i * 2 * stride],
                            wrk, &lwrk, iwrk1, &kwrk1, &err);
#if 0
                    bispev_f(knots_theta, nt, knots_phi, np, coeff, 3, 3,
                            &theta1, 1, &phi1, 1, &out[i * 2 * stride],
                            wrk, lwrk, iwrk1, kwrk1, &err);
#endif
                    if (err != 0) return OSKAR_ERR_SPLINE_EVAL_FAIL;
                }
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            double wrk[8];
            for (j = 0; j < 2; ++j)
            {
                const double *knots_theta, *knots_phi, *coeff;
                double* out;
                if (j == 0) /* Real part. */
                {
                    nt          = spline->num_knots_x_re;
                    np          = spline->num_knots_y_re;
                    knots_theta = (const double*)spline->knots_x_re.data;
                    knots_phi   = (const double*)spline->knots_y_re.data;
                    coeff       = (const double*)spline->coeff_re.data;
                    out         = (double*)output->data;
                }
                else  /* Imaginary part. */
                {
                    nt          = spline->num_knots_x_im;
                    np          = spline->num_knots_y_im;
                    knots_theta = (const double*)spline->knots_x_im.data;
                    knots_phi   = (const double*)spline->knots_y_im.data;
                    coeff       = (const double*)spline->coeff_im.data;
                    out         = (double*)output->data + 1;
                }

                for (i = 0; i < num_points; ++i)
                {
                    double theta1, phi1;
                    theta1 = ((const double*)theta->data)[i];
                    phi1 = ((const double*)phi->data)[i];
                    bispev_d(knots_theta, nt, knots_phi, np, coeff, 3, 3,
                            &theta1, 1, &phi1, 1, &out[i * 2 * stride],
                            wrk, lwrk, iwrk1, kwrk1, &err);
                    if (err != 0) return OSKAR_ERR_SPLINE_EVAL_FAIL;
                }
            }
        }
    }

    /* Check if data are in GPU memory. */
    else if (location == OSKAR_LOCATION_GPU)
    {
        /* TODO Implement spline evaluation on GPU. */
        return OSKAR_ERR_BAD_LOCATION;
    }

    return err;
}

#ifdef __cplusplus
}
#endif
