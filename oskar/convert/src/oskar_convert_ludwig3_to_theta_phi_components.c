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

#include "convert/oskar_convert_ludwig3_to_theta_phi_components.h"
#include "convert/oskar_convert_ludwig3_to_theta_phi_components_cuda.h"
#include "utility/oskar_device_utils.h"
#include "convert/private_convert_ludwig3_to_theta_phi_components_inline.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_ludwig3_to_theta_phi_components_f(int num_points,
        float2* h_theta, float2* v_phi, const float* phi, int stride)
{
    int i;
    for (i = 0; i < num_points; ++i)
    {
        oskar_convert_ludwig3_to_theta_phi_components_inline_f(
                &h_theta[i*stride], &v_phi[i*stride], phi[i]);
    }
}

/* Double precision. */
void oskar_convert_ludwig3_to_theta_phi_components_d(int num_points,
        double2* h_theta, double2* v_phi, const double* phi, int stride)
{
    int i;
    for (i = 0; i < num_points; ++i)
    {
        oskar_convert_ludwig3_to_theta_phi_components_inline_d(
                &h_theta[i*stride], &v_phi[i*stride], phi[i]);
    }
}

/* Wrapper. */
void oskar_convert_ludwig3_to_theta_phi_components(oskar_Mem* vec,
        int offset, int stride, int num_points, const oskar_Mem* phi,
        int* status)
{
    int type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that the vector component data is in matrix form. */
    if (!oskar_mem_is_matrix(vec))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Get data type and location. */
    type = oskar_mem_type(phi);
    location = oskar_mem_location(phi);

    /* Convert vector representation from Ludwig-3 to spherical. */
    if (type == OSKAR_SINGLE)
    {
        float2 *h_theta, *v_phi;
        const float *phi_;
        h_theta = oskar_mem_float2(vec, status) + offset;
        v_phi = oskar_mem_float2(vec, status) + offset + 1;
        phi_ = oskar_mem_float_const(phi, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_convert_ludwig3_to_theta_phi_components_cuda_f(num_points,
                    h_theta, v_phi, phi_, stride);
            oskar_device_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            oskar_convert_ludwig3_to_theta_phi_components_f(num_points,
                    h_theta, v_phi, phi_, stride);
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        double2 *h_theta, *v_phi;
        const double *phi_;
        h_theta = oskar_mem_double2(vec, status) + offset;
        v_phi = oskar_mem_double2(vec, status) + offset + 1;
        phi_ = oskar_mem_double_const(phi, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_convert_ludwig3_to_theta_phi_components_cuda_d(num_points,
                    h_theta, v_phi, phi_, stride);
            oskar_device_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            oskar_convert_ludwig3_to_theta_phi_components_d(num_points,
                    h_theta, v_phi, phi_, stride);
        }
    }
    else
        *status = OSKAR_ERR_BAD_DATA_TYPE;
}

#ifdef __cplusplus
}
#endif
