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
#include "math/oskar_spherical_spline_data_evaluate.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_spherical_spline_data_evaluate(oskar_Mem* output,
        const oskar_SphericalSplineData* spline, const oskar_Mem* theta,
        const oskar_Mem* phi)
{
    int err = 0, i, kwrk1, lwrk, nt, np, num_points, type, location;

    /* Check arrays are consistent. */
    num_points = output->private_num_elements;
    if (num_points != theta->private_num_elements ||
            num_points != phi->private_num_elements)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check type. */
    type = output->private_type;
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        return OSKAR_ERR_BAD_DATA_TYPE;
    if (type != theta->private_type || type != phi->private_type)
        return OSKAR_ERR_TYPE_MISMATCH;

    /* Check location. */
    location = output->private_location;
    if (location != spline->coeff.private_location ||
            location != spline->knots_theta.private_location ||
            location != spline->knots_phi.private_location ||
            location != theta->private_location ||
            location != phi->private_location)
        return OSKAR_ERR_BAD_LOCATION;

    /* Get common data. */
    nt = spline->num_knots_theta;
    np = spline->num_knots_phi;

    /* Check if data are in CPU memory. */
    if (location == OSKAR_LOCATION_CPU)
    {
        /* Set up workspace. */
        int iwrk1[2];
        kwrk1 = sizeof(iwrk1) / sizeof(int);

        if (type == OSKAR_SINGLE)
        {
            const float *theta_f, *phi_f;
            float wrk[16];
            lwrk = sizeof(wrk) / sizeof(float);
            theta_f = (const float*)theta->data;
            phi_f = (const float*)phi->data;
            for (i = 0; i < num_points; ++i)
            {
                float val, theta1, phi1;
                theta1 = theta_f[i];
                phi1 = phi_f[i];
                bispev_f((const float*)spline->knots_theta.data, nt,
                        (const float*)spline->knots_phi.data, np,
                        (const float*)spline->coeff.data, 3, 3,
                        &theta1, 1, &phi1, 1, &val,
                        wrk, lwrk, iwrk1, kwrk1, &err);
                if (err != 0) return OSKAR_ERR_SPLINE_EVAL_FAIL;
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            const double *theta_d, *phi_d;
            double wrk[16];
            lwrk = sizeof(wrk) / sizeof(double);
            theta_d = (const double*)theta->data;
            phi_d = (const double*)phi->data;
            for (i = 0; i < num_points; ++i)
            {
                double val, theta1, phi1;
                theta1 = theta_d[i];
                phi1 = phi_d[i];
                bispev_d((const double*)spline->knots_theta.data, nt,
                        (const double*)spline->knots_phi.data, np,
                        (const double*)spline->coeff.data, 3, 3,
                        &theta1, 1, &phi1, 1, &val,
                        wrk, lwrk, iwrk1, kwrk1, &err);
                if (err != 0) return OSKAR_ERR_SPLINE_EVAL_FAIL;
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
