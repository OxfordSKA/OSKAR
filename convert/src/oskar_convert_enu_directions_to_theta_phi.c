/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#include <oskar_convert_enu_directions_to_theta_phi.h>
#include <oskar_convert_enu_directions_to_theta_phi_cuda.h>
#include <oskar_cuda_check_error.h>
#include <private_convert_enu_directions_to_theta_phi_inline.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_enu_directions_to_theta_phi_f(
        const int num_points, const float* x, const float* y,
        const float* z, const float delta_phi, float* theta, float* phi)
{
    int i;
    for (i = 0; i < num_points; ++i)
    {
        oskar_convert_enu_directions_to_theta_phi_inline_f(x[i], y[i],
                z[i], delta_phi, &theta[i], &phi[i]);
    }
}

/* Double precision. */
void oskar_convert_enu_directions_to_theta_phi_d(
        const int num_points, const double* x, const double* y,
        const double* z, const double delta_phi, double* theta, double* phi)
{
    int i;
    for (i = 0; i < num_points; ++i)
    {
        oskar_convert_enu_directions_to_theta_phi_inline_d(x[i], y[i],
                z[i], delta_phi, &theta[i], &phi[i]);
    }
}

/* Wrapper. */
void oskar_convert_enu_directions_to_theta_phi(int num_points,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double delta_phi, oskar_Mem* theta, oskar_Mem* phi, int* status)
{
    int type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get data type and location. */
    type = oskar_mem_type(theta);
    location = oskar_mem_location(theta);

    /* Compute modified theta and phi coordinates. */
    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
        {
            oskar_convert_enu_directions_to_theta_phi_cuda_f(num_points,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status), (float)delta_phi,
                    oskar_mem_float(theta, status),
                    oskar_mem_float(phi, status));
            oskar_cuda_check_error(status);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_convert_enu_directions_to_theta_phi_cuda_d(num_points,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status), delta_phi,
                    oskar_mem_double(theta, status),
                    oskar_mem_double(phi, status));
            oskar_cuda_check_error(status);
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_convert_enu_directions_to_theta_phi_f(num_points,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status), (float)delta_phi,
                    oskar_mem_float(theta, status),
                    oskar_mem_float(phi, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_convert_enu_directions_to_theta_phi_d(num_points,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status), delta_phi,
                    oskar_mem_double(theta, status),
                    oskar_mem_double(phi, status));
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
