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

#include <oskar_evaluate_element_weights_dft.h>
#include <oskar_evaluate_element_weights_dft_cuda.h>
#include <oskar_cuda_check_error.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_element_weights_dft_f(float2* weights,
        const int num_elements, const float wavenumber, const float* x,
        const float* y, const float* z, const float x_beam,
        const float y_beam, const float z_beam)
{
    int i;

    /* Loop over elements. */
    for (i = 0; i < num_elements; ++i)
    {
        float cxi, cyi, czi, phase;
        float2 weight;

        /* Cache input data. */
        cxi = wavenumber * x[i];
        cyi = wavenumber * y[i];
        czi = wavenumber * z[i];

        /* Compute the geometric phase of the output direction. */
        phase =  cxi * x_beam;
        phase += cyi * y_beam;
        phase += czi * z_beam;
        weight.x = cosf(-phase);
        weight.y = sinf(-phase);

        /* Store result. */
        weights[i] = weight;
    }
}

/* Double precision. */
void oskar_evaluate_element_weights_dft_d(double2* weights,
        const int num_elements, const double wavenumber, const double* x,
        const double* y, const double* z, const double x_beam,
        const double y_beam, const double z_beam)
{
    int i;

    /* Loop over elements. */
    for (i = 0; i < num_elements; ++i)
    {
        double cxi, cyi, czi, phase;
        double2 weight;

        /* Cache input data. */
        cxi = wavenumber * x[i];
        cyi = wavenumber * y[i];
        czi = wavenumber * z[i];

        /* Compute the geometric phase of the output direction. */
        phase =  cxi * x_beam;
        phase += cyi * y_beam;
        phase += czi * z_beam;
        weight.x = cos(-phase);
        weight.y = sin(-phase);

        /* Store result. */
        weights[i] = weight;
    }
}

/* Wrapper. */
void oskar_evaluate_element_weights_dft(oskar_Mem* weights, int num_elements,
        double wavenumber, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, double x_beam, double y_beam, double z_beam,
        int* status)
{
    int type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check array dimensions are OK. */
    if ((int)oskar_mem_length(weights) < num_elements ||
            (int)oskar_mem_length(x) < num_elements ||
            (int)oskar_mem_length(y) < num_elements ||
            (int)oskar_mem_length(z) < num_elements)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check for location mismatch. */
    location = oskar_mem_location(weights);
    if (oskar_mem_location(x) != location ||
            oskar_mem_location(y) != location ||
            oskar_mem_location(z) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check types. */
    type = oskar_mem_precision(weights);
    if (!oskar_mem_is_complex(weights) || oskar_mem_is_matrix(weights))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (oskar_mem_type(x) != type || oskar_mem_type(y) != type ||
            oskar_mem_type(z) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Generate DFT weights: switch on type and location. */
    if (type == OSKAR_DOUBLE)
    {
        const double *x_, *y_, *z_;
        double2* weights_;
        x_ = oskar_mem_double_const(x, status);
        y_ = oskar_mem_double_const(y, status);
        z_ = oskar_mem_double_const(z, status);
        weights_ = oskar_mem_double2(weights, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_element_weights_dft_cuda_d(weights_, num_elements,
                    wavenumber, x_, y_, z_, x_beam, y_beam, z_beam);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            oskar_evaluate_element_weights_dft_d(weights_, num_elements,
                    wavenumber, x_, y_, z_, x_beam, y_beam, z_beam);
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        const float *x_, *y_, *z_;
        float2* weights_;
        x_ = oskar_mem_float_const(x, status);
        y_ = oskar_mem_float_const(y, status);
        z_ = oskar_mem_float_const(z, status);
        weights_ = oskar_mem_float2(weights, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_element_weights_dft_cuda_f(weights_, num_elements,
                    (float)wavenumber, x_, y_, z_, (float)x_beam,
                    (float)y_beam, (float)z_beam);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            oskar_evaluate_element_weights_dft_f(weights_, num_elements,
                    (float)wavenumber, x_, y_, z_, (float)x_beam,
                    (float)y_beam, (float)z_beam);
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
