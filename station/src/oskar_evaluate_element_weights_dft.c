/*
 * Copyright (c) 2013, The University of Oxford
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

#include "station/oskar_evaluate_element_weights_dft.h"
#include "station/oskar_evaluate_element_weights_dft_cuda.h"
#include "utility/oskar_mem_type_check.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_element_weights_dft_f(float2* weights,
        const int num_elements, const float* x, const float* y,
        const float* z, const float x_beam, const float y_beam,
        const float z_beam)
{
    int i;

    /* Loop over elements. */
    for (i = 0; i < num_elements; ++i)
    {
        float cxi, cyi, czi, phase;
        float2 weight;

        /* Cache input data. */
        cxi = x[i];
        cyi = y[i];
        czi = z[i];

        /* Compute the geometric phase of the output direction. */
        phase =  cxi * x_beam;
        phase += cyi * y_beam;
        phase += czi * z_beam;
        weight.x = cosf(phase);
        weight.y = sinf(phase);

        /* Store result. */
        weights[i] = weight;
    }
}

/* Double precision. */
void oskar_evaluate_element_weights_dft_d(double2* weights,
        const int num_elements, const double* x, const double* y,
        const double* z, const double x_beam, const double y_beam,
        const double z_beam)
{
    int i;

    /* Loop over elements. */
    for (i = 0; i < num_elements; ++i)
    {
        double cxi, cyi, czi, phase;
        double2 weight;

        /* Cache input data. */
        cxi = x[i];
        cyi = y[i];
        czi = z[i];

        /* Compute the geometric phase of the output direction. */
        phase =  cxi * x_beam;
        phase += cyi * y_beam;
        phase += czi * z_beam;
        weight.x = cos(phase);
        weight.y = sin(phase);

        /* Store result. */
        weights[i] = weight;
    }
}

/* Wrapper. */
void oskar_evaluate_element_weights_dft(oskar_Mem* weights, int num_elements,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double x_beam, double y_beam, double z_beam, int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!weights || !x || !y || !z || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check array dimensions are OK. */
    if (weights->num_elements < num_elements ||
            x->num_elements < num_elements ||
            y->num_elements < num_elements ||
            z->num_elements < num_elements)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check for location mismatch. */
    location = weights->location;
    if (x->location != location ||
            y->location != location ||
            z->location != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check types. */
    type = oskar_mem_base_type(weights->type);
    if (!oskar_mem_is_complex(weights->type) ||
            oskar_mem_is_matrix(weights->type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (x->type != type || y->type != type || z->type != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Generate DFT weights. */
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_element_weights_dft_cuda_d
            ((double2*)weights->data, num_elements, (double*)x->data,
                    (double*)y->data, (double*)z->data, x_beam, y_beam,
                    z_beam);
        }
        else if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_element_weights_dft_cuda_f
            ((float2*)weights->data, num_elements, (float*)x->data,
                    (float*)y->data, (float*)z->data, x_beam, y_beam,
                    z_beam);
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
    {
        if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_element_weights_dft_d
            ((double2*)weights->data, num_elements, (double*)x->data,
                    (double*)y->data, (double*)z->data, x_beam, y_beam,
                    z_beam);
        }
        else if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_element_weights_dft_f
            ((float2*)weights->data, num_elements, (float*)x->data,
                    (float*)y->data, (float*)z->data, x_beam, y_beam,
                    z_beam);
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
}

#ifdef __cplusplus
}
#endif
