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

#include "extern/dierckx/bispev.h"
#include "math/oskar_spline_data_evaluate.h"
#include "math/cudak/oskar_cudak_dierckx_bispev_bicubic.h"
#include "utility/oskar_mem_type_check.h"

#define USE_FORTRAN_BISPEV 1

#ifdef __cplusplus
extern "C" {
#endif

#ifdef USE_FORTRAN_BISPEV
/* Fortran function prototype. */
void bispev_(const float tx[], const int* nx, const float ty[], const int* ny,
        const float c[], const int* kx, const int* ky, const float x[],
        const int* mx, const float y[], const int* my, float z[],
        float wrk[], const int* lwrk, int iwrk[], const int* kwrk, int* ier);

static int c1 = 1;
static int c3 = 3;
#endif

int oskar_spline_data_evaluate(oskar_Mem* output, int offset, int stride,
        const oskar_SplineData* spline, const oskar_Mem* x, const oskar_Mem* y)
{
    int err = 0, j, nx, ny, num_points, type, location;

    /* Check arrays are consistent. */
    num_points = x->num_elements;
    if (y->num_elements != num_points)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check type. */
    type = x->type;
    if (type != y->type)
        return OSKAR_ERR_TYPE_MISMATCH;
    if (!oskar_mem_is_complex(output->type))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check location. */
    location = output->location;
    if (location != spline->coeff_re.location ||
            location != spline->knots_x_re.location ||
            location != spline->knots_y_re.location ||
            location != x->location ||
            location != y->location)
        return OSKAR_ERR_BAD_LOCATION;

    /* Check data type. */
    if (type == OSKAR_SINGLE)
    {
        for (j = 0; j < 2; ++j)
        {
            const float *knots_x, *knots_y, *coeff;
            float *out;
            if (j == 0) /* Real part. */
            {
                nx      = spline->num_knots_x_re;
                ny      = spline->num_knots_y_re;
                knots_x = (const float*)spline->knots_x_re.data;
                knots_y = (const float*)spline->knots_y_re.data;
                coeff   = (const float*)spline->coeff_re.data;
                out     = (float*)output->data + (2 * offset);
            }
            else /* Imaginary part. */
            {
                nx      = spline->num_knots_x_im;
                ny      = spline->num_knots_y_im;
                knots_x = (const float*)spline->knots_x_im.data;
                knots_y = (const float*)spline->knots_y_im.data;
                coeff   = (const float*)spline->coeff_im.data;
                out     = (float*)output->data + (2 * offset) + 1;
            }

            /* Check if data are in CPU memory. */
            if (location == OSKAR_LOCATION_CPU)
            {
                /* Set up workspace. */
                float wrk[8];
                int i, iwrk1[2], kwrk1 = 2, lwrk = 8;

                /* Evaluate surface at the points. */
                for (i = 0; i < num_points; ++i)
                {
                    float x1, y1;
                    x1 = ((const float*)x->data)[i];
                    y1 = ((const float*)y->data)[i];
#ifdef USE_FORTRAN_BISPEV
                    bispev_(knots_x, &nx, knots_y, &ny, coeff, &c3, &c3,
                            &x1, &c1, &y1, &c1, &out[i * 2 * stride],
                            wrk, &lwrk, iwrk1, &kwrk1, &err);
#else
                    bispev_f(knots_x, nx, knots_y, ny, coeff, 3, 3,
                            &x1, 1, &y1, 1, &out[i * 2 * stride],
                            wrk, lwrk, iwrk1, kwrk1, &err);
#endif
                    if (err != 0) return OSKAR_ERR_SPLINE_EVAL_FAIL;
                }
            }
            else if (location == OSKAR_LOCATION_GPU)
            {
                /* Evaluate surface at the points by calling kernel. */
                int num_blocks, num_threads = 256;
                num_blocks = (num_points + num_threads - 1) / num_threads;
                oskar_cudak_dierckx_bispev_bicubic_f
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (knots_x,
                        nx, knots_y, ny, coeff, num_points,
                        (const float*)x->data, (const float*)y->data,
                        stride, out);
            }
            else
                return OSKAR_ERR_BAD_LOCATION;
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        for (j = 0; j < 2; ++j)
        {
            const double *knots_x, *knots_y, *coeff;
            double* out;
            if (j == 0) /* Real part. */
            {
                nx      = spline->num_knots_x_re;
                ny      = spline->num_knots_y_re;
                knots_x = (const double*)spline->knots_x_re.data;
                knots_y = (const double*)spline->knots_y_re.data;
                coeff   = (const double*)spline->coeff_re.data;
                out     = (double*)output->data + (2 * offset);
            }
            else /* Imaginary part. */
            {
                nx      = spline->num_knots_x_im;
                ny      = spline->num_knots_y_im;
                knots_x = (const double*)spline->knots_x_im.data;
                knots_y = (const double*)spline->knots_y_im.data;
                coeff   = (const double*)spline->coeff_im.data;
                out     = (double*)output->data + (2 * offset) + 1;
            }

            /* Check if data are in CPU memory. */
            if (location == OSKAR_LOCATION_CPU)
            {
                /* Set up workspace. */
                double wrk[8];
                int i, iwrk1[2], kwrk1 = 2, lwrk = 8;

                /* Evaluate surface at the points. */
                for (i = 0; i < num_points; ++i)
                {
                    double x1, y1;
                    x1 = ((const double*)x->data)[i];
                    y1 = ((const double*)y->data)[i];
                    bispev_d(knots_x, nx, knots_y, ny, coeff, 3, 3,
                            &x1, 1, &y1, 1, &out[i * 2 * stride],
                            wrk, lwrk, iwrk1, kwrk1, &err);
                    if (err != 0) return OSKAR_ERR_SPLINE_EVAL_FAIL;
                }
            }
            else if (location == OSKAR_LOCATION_GPU)
            {
                /* Evaluate surface at the points by calling kernel. */
                int num_blocks, num_threads = 256;
                num_blocks = (num_points + num_threads - 1) / num_threads;
                oskar_cudak_dierckx_bispev_bicubic_d
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (knots_x,
                        nx, knots_y, ny, coeff, num_points,
                        (const double*)x->data, (const double*)y->data,
                        stride, out);
            }
            else
                return OSKAR_ERR_BAD_LOCATION;
        }
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    if (location == OSKAR_LOCATION_GPU)
    {
        cudaDeviceSynchronize();
        err = (int) cudaPeekAtLastError();
    }
    return err;
}

#ifdef __cplusplus
}
#endif
