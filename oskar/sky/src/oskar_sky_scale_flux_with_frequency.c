/*
 * Copyright (c) 2011-2018, The University of Oxford
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

#include "sky/oskar_sky.h"
#include "sky/oskar_sky_scale_flux_with_frequency_cuda.h"
#include "sky/private_sky_scale_flux_with_frequency_inline.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_device_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_sky_scale_flux_with_frequency_f(int num_sources, float frequency,
        float* I, float* Q, float* U, float* V, float* ref_freq,
        const float* sp_index, const float* rm)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        oskar_scale_flux_with_frequency_inline_f(frequency,
                &I[i], &Q[i], &U[i], &V[i], &ref_freq[i], sp_index[i], rm[i]);
    }
}

/* Double precision. */
void oskar_sky_scale_flux_with_frequency_d(int num_sources, double frequency,
        double* I, double* Q, double* U, double* V, double* ref_freq,
        const double* sp_index, const double* rm)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        oskar_scale_flux_with_frequency_inline_d(frequency,
                &I[i], &Q[i], &U[i], &V[i], &ref_freq[i], sp_index[i], rm[i]);
    }
}

/* Wrapper. */
void oskar_sky_scale_flux_with_frequency(oskar_Sky* sky, double frequency,
        int* status)
{
    int type, location, num_sources;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the type, location and dimensions. */
    type = oskar_sky_precision(sky);
    location = oskar_sky_mem_location(sky);
    num_sources = oskar_sky_num_sources(sky);

    /* Scale the flux values. */
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
            oskar_sky_scale_flux_with_frequency_f(num_sources, frequency,
                    oskar_mem_float(oskar_sky_I(sky), status),
                    oskar_mem_float(oskar_sky_Q(sky), status),
                    oskar_mem_float(oskar_sky_U(sky), status),
                    oskar_mem_float(oskar_sky_V(sky), status),
                    oskar_mem_float(oskar_sky_reference_freq_hz(sky), status),
                    oskar_mem_float_const(
                            oskar_sky_spectral_index_const(sky), status),
                    oskar_mem_float_const(
                            oskar_sky_rotation_measure_rad_const(sky), status));
        else if (type == OSKAR_DOUBLE)
            oskar_sky_scale_flux_with_frequency_d(num_sources, frequency,
                    oskar_mem_double(oskar_sky_I(sky), status),
                    oskar_mem_double(oskar_sky_Q(sky), status),
                    oskar_mem_double(oskar_sky_U(sky), status),
                    oskar_mem_double(oskar_sky_V(sky), status),
                    oskar_mem_double(oskar_sky_reference_freq_hz(sky), status),
                    oskar_mem_double_const(
                            oskar_sky_spectral_index_const(sky), status),
                    oskar_mem_double_const(
                            oskar_sky_rotation_measure_rad_const(sky), status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
            oskar_sky_scale_flux_with_frequency_cuda_f(num_sources, frequency,
                    oskar_mem_float(oskar_sky_I(sky), status),
                    oskar_mem_float(oskar_sky_Q(sky), status),
                    oskar_mem_float(oskar_sky_U(sky), status),
                    oskar_mem_float(oskar_sky_V(sky), status),
                    oskar_mem_float(oskar_sky_reference_freq_hz(sky), status),
                    oskar_mem_float_const(
                            oskar_sky_spectral_index_const(sky), status),
                    oskar_mem_float_const(
                            oskar_sky_rotation_measure_rad_const(sky), status));
        else if (type == OSKAR_DOUBLE)
            oskar_sky_scale_flux_with_frequency_cuda_d(num_sources, frequency,
                    oskar_mem_double(oskar_sky_I(sky), status),
                    oskar_mem_double(oskar_sky_Q(sky), status),
                    oskar_mem_double(oskar_sky_U(sky), status),
                    oskar_mem_double(oskar_sky_V(sky), status),
                    oskar_mem_double(oskar_sky_reference_freq_hz(sky), status),
                    oskar_mem_double_const(
                            oskar_sky_spectral_index_const(sky), status),
                    oskar_mem_double_const(
                            oskar_sky_rotation_measure_rad_const(sky), status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        oskar_device_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        cl_event event;
        cl_kernel k = 0;
        cl_int error, num;
        cl_uint arg = 0;
        size_t global_size, local_size;
        if (type == OSKAR_DOUBLE)
            k = oskar_cl_kernel("scale_flux_with_frequency_double");
        else if (type == OSKAR_SINGLE)
            k = oskar_cl_kernel("scale_flux_with_frequency_float");
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        if (!k)
        {
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            return;
        }

        /* Set kernel arguments. */
        num = (cl_int) num_sources;
        error = clSetKernelArg(k, arg++, sizeof(cl_int), &num);
        if (type == OSKAR_SINGLE)
        {
            const cl_float freq = (cl_float) frequency;
            error |= clSetKernelArg(k, arg++, sizeof(cl_float), &freq);
        }
        else if (type == OSKAR_DOUBLE)
        {
            const cl_double freq = (cl_double) frequency;
            error |= clSetKernelArg(k, arg++, sizeof(cl_double), &freq);
        }
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer(oskar_sky_I(sky), status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer(oskar_sky_Q(sky), status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer(oskar_sky_U(sky), status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer(oskar_sky_V(sky), status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer(oskar_sky_reference_freq_hz(sky), status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(oskar_sky_spectral_index_const(sky), status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(oskar_sky_rotation_measure_rad_const(sky), status));
        if (error != CL_SUCCESS)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            return;
        }

        /* Launch kernel on current command queue. */
        local_size = oskar_cl_is_gpu() ? 256 : 128;
        global_size = ((num + local_size - 1) / local_size) * local_size;
        error = clEnqueueNDRangeKernel(oskar_cl_command_queue(), k, 1, NULL,
                &global_size, &local_size, 0, NULL, &event);
        if (error != CL_SUCCESS)
            *status = OSKAR_ERR_KERNEL_LAUNCH_FAILURE;
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
