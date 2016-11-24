/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include "splines/oskar_dierckx_bispev.h"
#include "splines/oskar_dierckx_bispev_bicubic_cuda.h"
#include "splines/oskar_splines.h"
#include "utility/oskar_device_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_splines_evaluate(oskar_Mem* output, int offset, int stride,
        const oskar_Splines* spline, int num_points, const oskar_Mem* x,
        const oskar_Mem* y, int* status)
{
    int nx, ny, type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type. */
    type = oskar_splines_precision(spline);
    if (type != oskar_mem_type(x) || type != oskar_mem_type(y))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check that all arrays are co-located. */
    location = oskar_splines_mem_location(spline);
    if (location != oskar_mem_location(output) ||
            location != oskar_mem_location(x) ||
            location != oskar_mem_location(y))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Get number of knots. */
    nx = oskar_splines_num_knots_x_theta(spline);
    ny = oskar_splines_num_knots_y_phi(spline);

    /* Check data type. */
    if (type == OSKAR_SINGLE)
    {
        const float *tx, *ty, *coeff, *x_, *y_;
        float *out;
        tx    = oskar_mem_float_const(
                oskar_splines_knots_x_theta_const(spline), status);
        ty    = oskar_mem_float_const(
                oskar_splines_knots_y_phi_const(spline), status);
        coeff = oskar_mem_float_const(
                oskar_splines_coeff_const(spline), status);
        x_    = oskar_mem_float_const(x, status);
        y_    = oskar_mem_float_const(y, status);
        out   = oskar_mem_float(output, status) + offset;

        /* Check if data are in CPU memory. */
        if (location == OSKAR_CPU)
        {
            int i;
            if (nx == 0 || ny == 0 || !tx || !ty || !coeff)
            {
                for (i = 0; i < num_points; ++i)
                {
                    out[i * stride] = 0.0f;
                }
            }
            else
            {
                /* Set up workspace. */
                float x1, y1, wrk[8];
                int iwrk1[2], kwrk1 = 2, lwrk = 8, err = 0;

                /* Evaluate surface at the points. */
                for (i = 0; i < num_points; ++i)
                {
                    x1 = x_[i];
                    y1 = y_[i];
                    oskar_dierckx_bispev_f(tx, nx, ty, ny, coeff, 3, 3,
                            &x1, 1, &y1, 1, &out[i * stride],
                            wrk, lwrk, iwrk1, kwrk1, &err);
                    if (err != 0)
                    {
                        *status = OSKAR_ERR_SPLINE_EVAL_FAIL;
                        return;
                    }
                }
            }
        }
        else if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_dierckx_bispev_bicubic_cuda_f(tx, nx, ty, ny, coeff,
                    num_points, x_, y_, stride, out);
            oskar_device_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
    }
    else if (type == OSKAR_DOUBLE)
    {
        const double *tx, *ty, *coeff, *x_, *y_;
        double *out;
        tx    = oskar_mem_double_const(
                oskar_splines_knots_x_theta_const(spline), status);
        ty    = oskar_mem_double_const(
                oskar_splines_knots_y_phi_const(spline), status);
        coeff = oskar_mem_double_const(
                oskar_splines_coeff_const(spline), status);
        x_    = oskar_mem_double_const(x, status);
        y_    = oskar_mem_double_const(y, status);
        out   = oskar_mem_double(output, status) + offset;

        /* Check if data are in CPU memory. */
        if (location == OSKAR_CPU)
        {
            int i;
            if (nx == 0 || ny == 0 || !tx || !ty || !coeff)
            {
                for (i = 0; i < num_points; ++i)
                {
                    out[i * stride] = 0.0;
                }
            }
            else
            {
                /* Set up workspace. */
                double x1, y1, wrk[8];
                int iwrk1[2], kwrk1 = 2, lwrk = 8, err = 0;

                /* Evaluate surface at the points. */
                for (i = 0; i < num_points; ++i)
                {
                    x1 = x_[i];
                    y1 = y_[i];
                    oskar_dierckx_bispev_d(tx, nx, ty, ny, coeff, 3, 3,
                            &x1, 1, &y1, 1, &out[i * stride],
                            wrk, lwrk, iwrk1, kwrk1, &err);
                    if (err != 0)
                    {
                        *status = OSKAR_ERR_SPLINE_EVAL_FAIL;
                        return;
                    }
                }
            }
        }
        else if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_dierckx_bispev_bicubic_cuda_d(tx, nx, ty, ny, coeff,
                    num_points, x_, y_, stride, out);
            oskar_device_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
    }
    else
        *status = OSKAR_ERR_BAD_DATA_TYPE;
}

#ifdef __cplusplus
}
#endif
