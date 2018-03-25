/*
 * Copyright (c) 2012-2018, The University of Oxford
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

#include "telescope/station/element/oskar_apply_element_taper_gaussian.h"
#include "telescope/station/element/oskar_apply_element_taper_gaussian_cuda.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_device_utils.h"
#include <math.h>

#define M_4LN2  2.77258872223978123767

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_apply_element_taper_gaussian_scalar_f(const int num_sources,
        const float inv_2sigma_sq, const float* theta, float2* jones)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        float f, theta_sq;
        theta_sq = theta[i];
        theta_sq *= theta_sq;
        f = expf(-theta_sq * inv_2sigma_sq);
        jones[i].x *= f;
        jones[i].y *= f;
    }
}

void oskar_apply_element_taper_gaussian_matrix_f(const int num_sources,
        const float inv_2sigma_sq, const float* theta, float4c* jones)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        float f, theta_sq;
        theta_sq = theta[i];
        theta_sq *= theta_sq;
        f = expf(-theta_sq * inv_2sigma_sq);
        jones[i].a.x *= f;
        jones[i].a.y *= f;
        jones[i].b.x *= f;
        jones[i].b.y *= f;
        jones[i].c.x *= f;
        jones[i].c.y *= f;
        jones[i].d.x *= f;
        jones[i].d.y *= f;
    }
}

/* Double precision. */
void oskar_apply_element_taper_gaussian_scalar_d(const int num_sources,
        const double inv_2sigma_sq, const double* theta, double2* jones)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        double f, theta_sq;
        theta_sq = theta[i];
        theta_sq *= theta_sq;
        f = exp(-theta_sq * inv_2sigma_sq);
        jones[i].x *= f;
        jones[i].y *= f;
    }
}

void oskar_apply_element_taper_gaussian_matrix_d(const int num_sources,
        const double inv_2sigma_sq, const double* theta, double4c* jones)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        double f, theta_sq;
        theta_sq = theta[i];
        theta_sq *= theta_sq;
        f = exp(-theta_sq * inv_2sigma_sq);
        jones[i].a.x *= f;
        jones[i].a.y *= f;
        jones[i].b.x *= f;
        jones[i].b.y *= f;
        jones[i].c.x *= f;
        jones[i].c.y *= f;
        jones[i].d.x *= f;
        jones[i].d.y *= f;
    }
}

/* Wrapper. */
void oskar_apply_element_taper_gaussian(oskar_Mem* jones, int num_sources,
        double fwhm, const oskar_Mem* theta, int* status)
{
    if (*status) return;

    /* Check arrays are co-located. */
    const int location = oskar_mem_location(jones);
    if (oskar_mem_location(theta) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check types for consistency. */
    if (oskar_mem_type(theta) != oskar_mem_precision(jones))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Apply taper. */
    const double inv_2sigma_sq = M_4LN2 / (fwhm * fwhm);
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(jones))
        {
        case OSKAR_SINGLE_COMPLEX:
            oskar_apply_element_taper_gaussian_scalar_f(num_sources,
                    (float)inv_2sigma_sq, oskar_mem_float_const(theta, status),
                    oskar_mem_float2(jones, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            oskar_apply_element_taper_gaussian_scalar_d(num_sources,
                    inv_2sigma_sq, oskar_mem_double_const(theta, status),
                    oskar_mem_double2(jones, status));
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            oskar_apply_element_taper_gaussian_matrix_f(num_sources,
                    (float)inv_2sigma_sq, oskar_mem_float_const(theta, status),
                    oskar_mem_float4c(jones, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            oskar_apply_element_taper_gaussian_matrix_d(num_sources,
                    inv_2sigma_sq, oskar_mem_double_const(theta, status),
                    oskar_mem_double4c(jones, status));
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        switch (oskar_mem_type(jones))
        {
        case OSKAR_SINGLE_COMPLEX:
            oskar_apply_element_taper_gaussian_scalar_cuda_f(num_sources,
                    (float)inv_2sigma_sq, oskar_mem_float_const(theta, status),
                    oskar_mem_float2(jones, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            oskar_apply_element_taper_gaussian_scalar_cuda_d(num_sources,
                    inv_2sigma_sq, oskar_mem_double_const(theta, status),
                    oskar_mem_double2(jones, status));
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            oskar_apply_element_taper_gaussian_matrix_cuda_f(num_sources,
                    (float)inv_2sigma_sq, oskar_mem_float_const(theta, status),
                    oskar_mem_float4c(jones, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            oskar_apply_element_taper_gaussian_matrix_cuda_d(num_sources,
                    inv_2sigma_sq, oskar_mem_double_const(theta, status),
                    oskar_mem_double4c(jones, status));
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        }
        oskar_device_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        cl_device_type dev_type;
        cl_event event;
        cl_kernel k = 0;
        cl_int is_gpu, error, num;
        cl_uint arg = 0;
        size_t global_size, local_size;
        clGetDeviceInfo(oskar_cl_device_id(),
                CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL);
        is_gpu = dev_type & CL_DEVICE_TYPE_GPU;
        switch (oskar_mem_type(jones))
        {
        case OSKAR_SINGLE_COMPLEX:
            k = oskar_cl_kernel("apply_element_taper_gaussian_scalar_float");
            break;
        case OSKAR_DOUBLE_COMPLEX:
            k = oskar_cl_kernel("apply_element_taper_gaussian_scalar_double");
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = oskar_cl_kernel("apply_element_taper_gaussian_matrix_float");
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = oskar_cl_kernel("apply_element_taper_gaussian_matrix_double");
            break;
        default:
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
        if (oskar_mem_precision(jones) == OSKAR_SINGLE)
        {
            cl_float w = (cl_float) inv_2sigma_sq;
            error = clSetKernelArg(k, arg++, sizeof(cl_float), &w);
        }
        else
        {
            cl_double w = (cl_double) inv_2sigma_sq;
            error = clSetKernelArg(k, arg++, sizeof(cl_double), &w);
        }
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(theta, status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer(jones, status));
        if (error != CL_SUCCESS)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            return;
        }

        /* Launch kernel on current command queue. */
        local_size = is_gpu ? 256 : 128;
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
