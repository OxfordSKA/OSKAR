/*
 * Copyright (c) 2012-2017, The University of Oxford
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

#include "mem/oskar_mem.h"
#include "mem/oskar_mem_set_value_real_cuda.h"
#include "mem/private_mem.h"
#include "utility/oskar_cl_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_set_value_real(oskar_Mem* mem, double val,
        size_t offset, size_t length, int* status)
{
    size_t i, n;
    int type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data type, location, and number of elements. */
    type = mem->type;
    location = mem->location;
    n = length;
    if (offset == 0 && length == 0)
    {
        n = mem->num_elements;
    }

    if (location == OSKAR_CPU)
    {
        switch (type)
        {
        case OSKAR_DOUBLE:
        {
            double *v;
            v = (double*)(mem->data) + offset;
            for (i = 0; i < n; ++i) v[i] = val;
            break;
        }
        case OSKAR_DOUBLE_COMPLEX:
        {
            double2 *v;
            v = (double2*)(mem->data) + offset;
            for (i = 0; i < n; ++i)
            {
                v[i].x = val;
                v[i].y = 0.0;
            }
            break;
        }
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
        {
            double4c d;
            double4c *v;
            v = (double4c*)(mem->data) + offset;
            for (i = 0; i < n; ++i)
            {
                d.a.x = val;
                d.a.y = 0.0;
                d.b.x = 0.0;
                d.b.y = 0.0;
                d.c.x = 0.0;
                d.c.y = 0.0;
                d.d.x = val;
                d.d.y = 0.0;
                v[i] = d;
            }
            break;
        }
        case OSKAR_SINGLE:
        {
            float *v;
            v = (float*)(mem->data) + offset;
            for (i = 0; i < n; ++i) v[i] = (float)val;
            break;
        }
        case OSKAR_SINGLE_COMPLEX:
        {
            float2 *v;
            v = (float2*)(mem->data) + offset;
            for (i = 0; i < n; ++i)
            {
                v[i].x = (float)val;
                v[i].y = 0.0f;
            }
            break;
        }
        case OSKAR_SINGLE_COMPLEX_MATRIX:
        {
            float4c d;
            float4c *v;
            v = (float4c*)(mem->data) + offset;
            for (i = 0; i < n; ++i)
            {
                d.a.x = (float)val;
                d.a.y = 0.0f;
                d.b.x = 0.0f;
                d.b.y = 0.0f;
                d.c.x = 0.0f;
                d.c.y = 0.0f;
                d.d.x = (float)val;
                d.d.y = 0.0f;
                v[i] = d;
            }
            break;
        }
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
        case OSKAR_DOUBLE:
            oskar_mem_set_value_real_cuda_r_d((int)n,
                    (double*)(mem->data) + offset, val);
            break;
        case OSKAR_DOUBLE_COMPLEX:
            oskar_mem_set_value_real_cuda_c_d((int)n,
                    (double2*)(mem->data) + offset, val);
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            oskar_mem_set_value_real_cuda_m_d((int)n,
                    (double4c*)(mem->data) + offset, val);
            break;
        case OSKAR_SINGLE:
            oskar_mem_set_value_real_cuda_r_f((int)n,
                    (float*)(mem->data) + offset, (float)val);
            break;
        case OSKAR_SINGLE_COMPLEX:
            oskar_mem_set_value_real_cuda_c_f((int)n,
                    (float2*)(mem->data) + offset, (float)val);
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            oskar_mem_set_value_real_cuda_m_f((int)n,
                    (float4c*)(mem->data) + offset, (float)val);
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        }
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
        cl_int is_gpu, error, n_in, off;
        cl_uint arg = 0;

        /* Get the appropriate kernel. */
        /*
         * NOTE: Don't use clEnqueueFillBuffer(),
         * as this is currently broken on macOS.
         */
        clGetDeviceInfo(oskar_cl_device_id(),
                CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL);
        is_gpu = dev_type & CL_DEVICE_TYPE_GPU;
        const size_t local_size = is_gpu ? 256 : 128;
        switch (type)
        {
        case OSKAR_DOUBLE:
            k = oskar_cl_kernel("mem_set_value_real_r_double");
            break;
        case OSKAR_DOUBLE_COMPLEX:
            k = oskar_cl_kernel("mem_set_value_real_c_double");
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = oskar_cl_kernel("mem_set_value_real_m_double");
            break;
        case OSKAR_SINGLE:
            k = oskar_cl_kernel("mem_set_value_real_r_float");
            break;
        case OSKAR_SINGLE_COMPLEX:
            k = oskar_cl_kernel("mem_set_value_real_c_float");
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = oskar_cl_kernel("mem_set_value_real_m_float");
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        }

        /* Set kernel arguments. */
        if (k)
        {
            n_in = (cl_int) (mem->num_elements);
            off = (cl_int) offset;
            error = clSetKernelArg(k, arg++, sizeof(cl_int), &n_in);
            error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                    oskar_mem_cl_buffer_const(mem, status));
            error |= clSetKernelArg(k, arg++, sizeof(cl_int), &off);
            if (oskar_mem_precision(mem) == OSKAR_SINGLE)
            {
                cl_float v = (cl_float) val;
                error |= clSetKernelArg(k, arg++, sizeof(cl_float), &v);
            }
            else
            {
                cl_double v = (cl_double) val;
                error |= clSetKernelArg(k, arg++, sizeof(cl_double), &v);
            }
            if (!*status && error != CL_SUCCESS)
                *status = OSKAR_ERR_INVALID_ARGUMENT;
            if (!*status)
            {
                /* Launch kernel on current command queue. */
                const size_t global_size = ((n + local_size - 1) /
                        local_size) * local_size;
                error = clEnqueueNDRangeKernel(oskar_cl_command_queue(), k, 1,
                        NULL, &global_size, &local_size, 0, NULL, &event);
                if (error != CL_SUCCESS)
                    *status = OSKAR_ERR_KERNEL_LAUNCH_FAILURE;
            }
        }
        else
        {
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        }
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
