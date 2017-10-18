/*
 * Copyright (c) 2015-2017, The University of Oxford
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
#include "mem/oskar_mem_random_uniform_cuda.h"
#include "math/private_random_helpers.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_device_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_random_uniform_f(
        const int num_elements, float* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3)
{
    int i, i4, n1;

    n1 = num_elements / 4;
#pragma omp parallel for private(i, i4)
    for (i = 0; i < n1; ++i)
    {
        OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

        /* Convert to uniform float. */
        i4 = i * 4;
        data[i4]     = oskar_int_to_range_0_to_1_f(u.i[0]);
        data[i4 + 1] = oskar_int_to_range_0_to_1_f(u.i[1]);
        data[i4 + 2] = oskar_int_to_range_0_to_1_f(u.i[2]);
        data[i4 + 3] = oskar_int_to_range_0_to_1_f(u.i[3]);
    }

    if (num_elements % 4)
    {
        OSKAR_R123_GENERATE_4(seed, n1, counter1, counter2, counter3)

        /* Convert to uniform float. */
        i4 = n1 * 4;
        data[i4]         = oskar_int_to_range_0_to_1_f(u.i[0]);
        if (i4 + 1 < num_elements)
            data[i4 + 1] = oskar_int_to_range_0_to_1_f(u.i[1]);
        if (i4 + 2 < num_elements)
            data[i4 + 2] = oskar_int_to_range_0_to_1_f(u.i[2]);
        if (i4 + 3 < num_elements)
            data[i4 + 3] = oskar_int_to_range_0_to_1_f(u.i[3]);
    }
}

void oskar_mem_random_uniform_d(
        const int num_elements, double* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3)
{
    int i, i4, n1;

    n1 = num_elements / 4;
#pragma omp parallel for private(i, i4)
    for (i = 0; i < n1; ++i)
    {
        OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

        /* Convert to uniform float. */
        i4 = i * 4;
        data[i4]     = oskar_int_to_range_0_to_1_d(u.i[0]);
        data[i4 + 1] = oskar_int_to_range_0_to_1_d(u.i[1]);
        data[i4 + 2] = oskar_int_to_range_0_to_1_d(u.i[2]);
        data[i4 + 3] = oskar_int_to_range_0_to_1_d(u.i[3]);
    }

    if (num_elements % 4)
    {
        OSKAR_R123_GENERATE_4(seed, n1, counter1, counter2, counter3)

        /* Convert to uniform float. */
        i4 = n1 * 4;
        data[i4]         = oskar_int_to_range_0_to_1_d(u.i[0]);
        if (i4 + 1 < num_elements)
            data[i4 + 1] = oskar_int_to_range_0_to_1_d(u.i[1]);
        if (i4 + 2 < num_elements)
            data[i4 + 2] = oskar_int_to_range_0_to_1_d(u.i[2]);
        if (i4 + 3 < num_elements)
            data[i4 + 3] = oskar_int_to_range_0_to_1_d(u.i[3]);
    }
}

void oskar_mem_random_uniform(oskar_Mem* data, unsigned int seed,
        unsigned int counter1, unsigned int counter2, unsigned int counter3,
        int* status)
{
    int type, location;
    size_t num_elements;
#ifdef OSKAR_HAVE_OPENCL
    cl_kernel k = 0;
#endif

    /* Check if safe to proceed. */
    if (*status) return;

    type = oskar_mem_precision(data);
    location = oskar_mem_location(data);
    num_elements = oskar_mem_length(data);
    if (oskar_mem_is_complex(data)) num_elements *= 2;
    if (oskar_mem_is_matrix(data)) num_elements *= 4;

    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
            oskar_mem_random_uniform_cuda_f((int)num_elements,
                    oskar_mem_float(data, status), seed,
                    counter1, counter2, counter3);
        else if (type == OSKAR_DOUBLE)
            oskar_mem_random_uniform_cuda_d((int)num_elements,
                    oskar_mem_double(data, status), seed,
                    counter1, counter2, counter3);
        oskar_device_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
            oskar_mem_random_uniform_f((int)num_elements,
                    oskar_mem_float(data, status), seed,
                    counter1, counter2, counter3);
        else if (type == OSKAR_DOUBLE)
            oskar_mem_random_uniform_d((int)num_elements,
                    oskar_mem_double(data, status), seed,
                    counter1, counter2, counter3);
    }
    else if (location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        if (type == OSKAR_SINGLE)
            k = oskar_cl_kernel("mem_random_uniform_float");
        else if (type == OSKAR_DOUBLE)
            k = oskar_cl_kernel("mem_random_uniform_double");

        if (k)
        {
            cl_device_type dev_type;
            cl_event event;
            cl_int error, gpu;
            cl_uint n, s, c1, c2, c3;
            size_t global_size, local_size;

            /* Set kernel arguments. */
            clGetDeviceInfo(oskar_cl_device_id(),
                    CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL);
            gpu = dev_type & CL_DEVICE_TYPE_GPU;
            n = (cl_uint) num_elements;
            s = (cl_uint) seed;
            c1 = (cl_uint) counter1;
            c2 = (cl_uint) counter2;
            c3 = (cl_uint) counter3;
            error = clSetKernelArg(k, 0, sizeof(cl_uint), &n);
            error |= clSetKernelArg(k, 1, sizeof(cl_mem),
                    oskar_mem_cl_buffer(data, status));
            error |= clSetKernelArg(k, 2, sizeof(cl_uint), &s);
            error |= clSetKernelArg(k, 3, sizeof(cl_uint), &c1);
            error |= clSetKernelArg(k, 4, sizeof(cl_uint), &c2);
            error |= clSetKernelArg(k, 5, sizeof(cl_uint), &c3);
            if (*status) return;
            if (error != CL_SUCCESS)
            {
                *status = OSKAR_ERR_INVALID_ARGUMENT;
                return;
            }

            /* Launch kernel on current command queue. */
            local_size = gpu ? 256 : 128;
            global_size = ((((num_elements + 3) / 4) + local_size - 1) /
                    local_size) * local_size;
            error = clEnqueueNDRangeKernel(oskar_cl_command_queue(), k, 1, NULL,
                        &global_size, &local_size, 0, NULL, &event);
            if (error != CL_SUCCESS)
            {
                *status = OSKAR_ERR_KERNEL_LAUNCH_FAILURE;
                return;
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
