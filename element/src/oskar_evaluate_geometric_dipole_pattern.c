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

#include <oskar_evaluate_geometric_dipole_pattern.h>
#include <oskar_evaluate_geometric_dipole_pattern_cuda.h>
#include <oskar_evaluate_geometric_dipole_pattern_inline.h>
#include <oskar_cuda_check_error.h>
#include <oskar_cmath.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_geometric_dipole_pattern_f(int num_points,
        const float* theta, const float* phi, int stride,
        float2* E_theta, float2* E_phi)
{
    int i, i_out;

    for (i = 0; i < num_points; ++i)
    {
        i_out = i * stride;
        oskar_evaluate_geometric_dipole_pattern_inline_f(theta[i], phi[i],
                E_theta + i_out, E_phi + i_out);
    }
}

void oskar_evaluate_geometric_dipole_pattern_scalar_f(int num_points,
        const float* theta, const float* phi, int stride, float2* pattern)
{
    float theta_, phi_, amp;
    float4c val;
    int i, i_out;

    for (i = 0; i < num_points; ++i)
    {
        /* Get source coordinates. */
        theta_ = theta[i];
        phi_ = phi[i];

        /* Evaluate E_theta, E_phi for both X and Y dipoles. */
        oskar_evaluate_geometric_dipole_pattern_inline_f(theta_,
                phi_, &val.a, &val.b);
        oskar_evaluate_geometric_dipole_pattern_inline_f(theta_,
                phi_ + ((float)M_PI) / 2.0f, &val.c, &val.d);

        /* Get sum of the diagonal of the autocorrelation matrix. */
        amp = val.a.x * val.a.x + val.a.y * val.a.y +
                val.b.x * val.b.x + val.b.y * val.b.y +
                val.c.x * val.c.x + val.c.y * val.c.y +
                val.d.x * val.d.x + val.d.y * val.d.y;
        amp = sqrtf(0.5f * amp);

        /* Save amplitude. */
        i_out = i * stride;
        pattern[i_out].x = amp;
        pattern[i_out].y = 0.0f;
    }
}

/* Double precision. */
void oskar_evaluate_geometric_dipole_pattern_d(int num_points,
        const double* theta, const double* phi, int stride,
        double2* E_theta, double2* E_phi)
{
    int i, i_out;

    for (i = 0; i < num_points; ++i)
    {
        i_out = i * stride;
        oskar_evaluate_geometric_dipole_pattern_inline_d(theta[i], phi[i],
                E_theta + i_out, E_phi + i_out);
    }
}

void oskar_evaluate_geometric_dipole_pattern_scalar_d(int num_points,
        const double* theta, const double* phi, int stride, double2* pattern)
{
    double theta_, phi_, amp;
    double4c val;
    int i, i_out;

    for (i = 0; i < num_points; ++i)
    {
        /* Get source coordinates. */
        theta_ = theta[i];
        phi_ = phi[i];

        /* Evaluate E_theta, E_phi for both X and Y dipoles. */
        oskar_evaluate_geometric_dipole_pattern_inline_d(theta_,
                phi_, &val.a, &val.b);
        oskar_evaluate_geometric_dipole_pattern_inline_d(theta_,
                phi_ + M_PI / 2.0, &val.c, &val.d);

        /* Get sum of the diagonal of the autocorrelation matrix. */
        amp = val.a.x * val.a.x + val.a.y * val.a.y +
                val.b.x * val.b.x + val.b.y * val.b.y +
                val.c.x * val.c.x + val.c.y * val.c.y +
                val.d.x * val.d.x + val.d.y * val.d.y;
        amp = sqrt(0.5 * amp);

        /* Save amplitude. */
        i_out = i * stride;
        pattern[i_out].x = amp;
        pattern[i_out].y = 0.0;
    }
}


/* Wrapper. */
void oskar_evaluate_geometric_dipole_pattern(oskar_Mem* pattern, int num_points,
        const oskar_Mem* theta, const oskar_Mem* phi, int offset, int stride,
        int* status)
{
    int precision, type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the meta-data. */
    precision = oskar_mem_precision(pattern);
    type = oskar_mem_type(pattern);
    location = oskar_mem_location(pattern);

    /* Check that all arrays are co-located. */
    if (oskar_mem_location(theta) != location ||
            oskar_mem_location(phi) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check that the pattern array is complex. */
    if (!oskar_mem_is_complex(pattern))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Check that the types match. */
    if (oskar_mem_type(theta) != precision || oskar_mem_type(phi) != precision)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check the location. */
    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            oskar_evaluate_geometric_dipole_pattern_cuda_f(num_points,
                    oskar_mem_float_const(theta, status),
                    oskar_mem_float_const(phi, status), stride,
                    oskar_mem_float2(pattern, status) + offset,
                    oskar_mem_float2(pattern, status) + offset + 1);
        }
        else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            oskar_evaluate_geometric_dipole_pattern_cuda_d(num_points,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status), stride,
                    oskar_mem_double2(pattern, status) + offset,
                    oskar_mem_double2(pattern, status) + offset + 1);
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            oskar_evaluate_geometric_dipole_pattern_scalar_cuda_f(num_points,
                    oskar_mem_float_const(theta, status),
                    oskar_mem_float_const(phi, status), stride,
                    oskar_mem_float2(pattern, status) + offset);
        }
        else if (type == OSKAR_DOUBLE_COMPLEX)
        {
            oskar_evaluate_geometric_dipole_pattern_scalar_cuda_d(num_points,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status), stride,
                    oskar_mem_double2(pattern, status) + offset);
        }
        oskar_cuda_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            oskar_evaluate_geometric_dipole_pattern_f(num_points,
                    oskar_mem_float_const(theta, status),
                    oskar_mem_float_const(phi, status), stride,
                    oskar_mem_float2(pattern, status) + offset,
                    oskar_mem_float2(pattern, status) + offset + 1);
        }
        else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            oskar_evaluate_geometric_dipole_pattern_d(num_points,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status), stride,
                    oskar_mem_double2(pattern, status) + offset,
                    oskar_mem_double2(pattern, status) + offset + 1);
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            oskar_evaluate_geometric_dipole_pattern_scalar_f(num_points,
                    oskar_mem_float_const(theta, status),
                    oskar_mem_float_const(phi, status), stride,
                    oskar_mem_float2(pattern, status) + offset);
        }
        else if (type == OSKAR_DOUBLE_COMPLEX)
        {
            oskar_evaluate_geometric_dipole_pattern_scalar_d(num_points,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status), stride,
                    oskar_mem_double2(pattern, status) + offset);
        }
    }
}

#ifdef __cplusplus
}
#endif
