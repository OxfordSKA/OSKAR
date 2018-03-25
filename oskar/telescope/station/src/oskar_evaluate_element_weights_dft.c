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

#include "telescope/station/oskar_evaluate_element_weights_dft.h"
#include "telescope/station/oskar_evaluate_element_weights_dft_cuda.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_device_utils.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_element_weights_dft_f(const int num_elements,
        const float* x, const float* y, const float* z,
        const float wavenumber, const float x_beam, const float y_beam,
        const float z_beam, float2* weights)
{
    int i;
    for (i = 0; i < num_elements; ++i)
    {
        float phase;
        float2 weight;
        phase = wavenumber * (x[i] * x_beam + y[i] * y_beam + z[i] * z_beam);
        weight.x = cosf(-phase);
        weight.y = sinf(-phase);
        weights[i] = weight;
    }
}

/* Double precision. */
void oskar_evaluate_element_weights_dft_d(const int num_elements,
        const double* x, const double* y, const double* z,
        const double wavenumber, const double x_beam, const double y_beam,
        const double z_beam, double2* weights)
{
    int i;
    for (i = 0; i < num_elements; ++i)
    {
        double phase;
        double2 weight;
        phase = wavenumber * (x[i] * x_beam + y[i] * y_beam + z[i] * z_beam);
        weight.x = cos(-phase);
        weight.y = sin(-phase);
        weights[i] = weight;
    }
}

/* Wrapper. */
void oskar_evaluate_element_weights_dft(int num_elements,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double wavenumber, double x_beam, double y_beam, double z_beam,
        oskar_Mem* weights, int* status)
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

    /* Generate DFT weights. */
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE)
            oskar_evaluate_element_weights_dft_d(num_elements,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status),
                    wavenumber, x_beam, y_beam, z_beam,
                    oskar_mem_double2(weights, status));
        else if (type == OSKAR_SINGLE)
            oskar_evaluate_element_weights_dft_f(num_elements,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status),
                    wavenumber, x_beam, y_beam, z_beam,
                    oskar_mem_float2(weights, status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_DOUBLE)
            oskar_evaluate_element_weights_dft_cuda_d(num_elements,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status),
                    wavenumber, x_beam, y_beam, z_beam,
                    oskar_mem_double2(weights, status));
        else if (type == OSKAR_SINGLE)
            oskar_evaluate_element_weights_dft_cuda_f(num_elements,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status),
                    wavenumber, x_beam, y_beam, z_beam,
                    oskar_mem_float2(weights, status));
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
        cl_device_type dev_type;
        cl_event event;
        cl_kernel k = 0;
        cl_int is_gpu, error, num;
        cl_uint arg = 0;
        size_t global_size, local_size;
        clGetDeviceInfo(oskar_cl_device_id(),
                CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL);
        is_gpu = dev_type & CL_DEVICE_TYPE_GPU;
        if (type == OSKAR_DOUBLE)
            k = oskar_cl_kernel("evaluate_element_weights_dft_double");
        else if (type == OSKAR_SINGLE)
            k = oskar_cl_kernel("evaluate_element_weights_dft_float");
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
        num = (cl_int) num_elements;
        error = clSetKernelArg(k, arg++, sizeof(cl_int), &num);
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(x, status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(y, status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(z, status));
        if (type == OSKAR_SINGLE)
        {
            const cl_float w = (cl_float) wavenumber;
            const cl_float x1 = (cl_float) x_beam;
            const cl_float y1 = (cl_float) y_beam;
            const cl_float z1 = (cl_float) z_beam;
            error |= clSetKernelArg(k, arg++, sizeof(cl_float), &w);
            error |= clSetKernelArg(k, arg++, sizeof(cl_float), &x1);
            error |= clSetKernelArg(k, arg++, sizeof(cl_float), &y1);
            error |= clSetKernelArg(k, arg++, sizeof(cl_float), &z1);
        }
        else if (type == OSKAR_DOUBLE)
        {
            const cl_double w = (cl_double) wavenumber;
            const cl_double x1 = (cl_double) x_beam;
            const cl_double y1 = (cl_double) y_beam;
            const cl_double z1 = (cl_double) z_beam;
            error |= clSetKernelArg(k, arg++, sizeof(cl_double), &w);
            error |= clSetKernelArg(k, arg++, sizeof(cl_double), &x1);
            error |= clSetKernelArg(k, arg++, sizeof(cl_double), &y1);
            error |= clSetKernelArg(k, arg++, sizeof(cl_double), &z1);
        }
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer(weights, status));
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
