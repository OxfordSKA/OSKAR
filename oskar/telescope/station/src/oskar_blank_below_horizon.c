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

#include "telescope/station/oskar_blank_below_horizon.h"
#include "telescope/station/oskar_blank_below_horizon_cuda.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_device_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_blank_below_horizon_matrix_f(const int num_sources,
        const float* restrict mask, float4c* restrict jones)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        if (mask[i] < 0.0f)
        {
            jones[i].a.x = 0.0f;
            jones[i].a.y = 0.0f;
            jones[i].b.x = 0.0f;
            jones[i].b.y = 0.0f;
            jones[i].c.x = 0.0f;
            jones[i].c.y = 0.0f;
            jones[i].d.x = 0.0f;
            jones[i].d.y = 0.0f;
        }
    }
}

void oskar_blank_below_horizon_scalar_f(const int num_sources,
        const float* restrict mask, float2* restrict jones)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        if (mask[i] < 0.0f)
        {
            jones[i].x = 0.0f;
            jones[i].y = 0.0f;
        }
    }
}

/* Double precision. */
void oskar_blank_below_horizon_matrix_d(const int num_sources,
        const double* restrict mask, double4c* restrict jones)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        if (mask[i] < 0.0)
        {
            jones[i].a.x = 0.0;
            jones[i].a.y = 0.0;
            jones[i].b.x = 0.0;
            jones[i].b.y = 0.0;
            jones[i].c.x = 0.0;
            jones[i].c.y = 0.0;
            jones[i].d.x = 0.0;
            jones[i].d.y = 0.0;
        }
    }
}

void oskar_blank_below_horizon_scalar_d(const int num_sources,
        const double* restrict mask, double2* restrict jones)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        if (mask[i] < 0.0)
        {
            jones[i].x = 0.0;
            jones[i].y = 0.0;
        }
    }
}


/* Wrapper. */
void oskar_blank_below_horizon(const int num_sources,
        const oskar_Mem* mask, oskar_Mem* data, int* status)
{
    int precision, type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that all arrays are co-located. */
    location = oskar_mem_location(data);
    if (oskar_mem_location(mask) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check that the mask type is OK. */
    precision = oskar_mem_type(mask);
    type = oskar_mem_type(data);
    if (precision != OSKAR_SINGLE && precision != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Check that the dimensions are OK. */
    if ((int)oskar_mem_length(data) < num_sources)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Zero the value of any positions below the horizon. */
    if (location == OSKAR_CPU)
    {
        switch (type)
        {
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            oskar_blank_below_horizon_matrix_d(num_sources,
                    oskar_mem_double_const(mask, status),
                    oskar_mem_double4c(data, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            oskar_blank_below_horizon_scalar_d(num_sources,
                    oskar_mem_double_const(mask, status),
                    oskar_mem_double2(data, status));
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            oskar_blank_below_horizon_matrix_f(num_sources,
                    oskar_mem_float_const(mask, status),
                    oskar_mem_float4c(data, status));
            break;
        case OSKAR_SINGLE_COMPLEX:
            oskar_blank_below_horizon_scalar_f(num_sources,
                    oskar_mem_float_const(mask, status),
                    oskar_mem_float2(data, status));
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        switch (type)
        {
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            oskar_blank_below_horizon_matrix_cuda_d(num_sources,
                    oskar_mem_double_const(mask, status),
                    oskar_mem_double4c(data, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            oskar_blank_below_horizon_scalar_cuda_d(num_sources,
                    oskar_mem_double_const(mask, status),
                    oskar_mem_double2(data, status));
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            oskar_blank_below_horizon_matrix_cuda_f(num_sources,
                    oskar_mem_float_const(mask, status),
                    oskar_mem_float4c(data, status));
            break;
        case OSKAR_SINGLE_COMPLEX:
            oskar_blank_below_horizon_scalar_cuda_f(num_sources,
                    oskar_mem_float_const(mask, status),
                    oskar_mem_float2(data, status));
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
        switch (type)
        {
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = oskar_cl_kernel("blank_below_horizon_matrix_double");
            break;
        case OSKAR_DOUBLE_COMPLEX:
            k = oskar_cl_kernel("blank_below_horizon_scalar_double");
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = oskar_cl_kernel("blank_below_horizon_matrix_float");
            break;
        case OSKAR_SINGLE_COMPLEX:
            k = oskar_cl_kernel("blank_below_horizon_scalar_float");
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
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(mask, status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer(data, status));
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
}

#ifdef __cplusplus
}
#endif
