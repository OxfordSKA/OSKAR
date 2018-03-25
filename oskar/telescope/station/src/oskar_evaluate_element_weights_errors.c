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

#include "telescope/station/oskar_evaluate_element_weights_errors.h"
#include "telescope/station/oskar_evaluate_element_weights_errors_cuda.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_device_utils.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_element_weights_errors_f(int num_elements,
        const float* amp_gain, const float* amp_error,
        const float* phase_offset, const float* phase_error,
        float2* errors)
{
    int i;

    for (i = 0; i < num_elements; ++i)
    {
        float2 r, t;

        /* Get two random numbers from a normalised Gaussian distribution. */
        r = errors[i];

        /* Evaluate the real and imaginary components of the error weight
         * for the antenna. */
        r.x *= amp_error[i];
        r.x += amp_gain[i]; /* Amplitude. */
        r.y *= phase_error[i];
        r.y += phase_offset[i]; /* Phase. */
        t.x = cosf(r.y);
        t.y = sinf(r.y);
        t.x *= r.x; /* Real. */
        t.y *= r.x; /* Imaginary. */
        errors[i] = t; /* Store. */
    }
}

/* Double precision. */
void oskar_evaluate_element_weights_errors_d(int num_elements,
        const double* amp_gain, const double* amp_error,
        const double* phase_offset, const double* phase_error,
        double2* errors)
{
    int i;

    for (i = 0; i < num_elements; ++i)
    {
        double2 r, t;

        /* Get two random numbers from a normalised Gaussian distribution. */
        r = errors[i];

        /* Evaluate the real and imaginary components of the error weight
         * for the antenna. */
        r.x *= amp_error[i];
        r.x += amp_gain[i]; /* Amplitude. */
        r.y *= phase_error[i];
        r.y += phase_offset[i]; /* Phase. */
        t.x = cos(r.y);
        t.y = sin(r.y);
        t.x *= r.x; /* Real. */
        t.y *= r.x; /* Imaginary. */
        errors[i] = t; /* Store. */
    }
}

/* Wrapper. */
void oskar_evaluate_element_weights_errors(int num_elements,
        const oskar_Mem* gain, const oskar_Mem* gain_error,
        const oskar_Mem* phase, const oskar_Mem* phase_error,
        unsigned int random_seed, int time_index, int station_id,
        oskar_Mem* errors, int* status)
{
    int type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check array dimensions are OK. */
    if ((int)oskar_mem_length(errors) < num_elements ||
            (int)oskar_mem_length(gain) < num_elements ||
            (int)oskar_mem_length(gain_error) < num_elements ||
            (int)oskar_mem_length(phase) < num_elements ||
            (int)oskar_mem_length(phase_error) < num_elements)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check for location mismatch. */
    location = oskar_mem_location(errors);
    if (oskar_mem_location(gain) != location ||
            oskar_mem_location(gain_error) != location ||
            oskar_mem_location(phase) != location ||
            oskar_mem_location(phase_error) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check types. */
    type = oskar_mem_precision(errors);
    if (!oskar_mem_is_complex(errors) || oskar_mem_is_matrix(errors))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (oskar_mem_type(gain) != type || oskar_mem_type(phase) != type ||
            oskar_mem_type(gain_error) != type ||
            oskar_mem_type(phase_error) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Generate Gaussian-distributed random numbers in output array. */
    oskar_mem_random_gaussian(errors, random_seed, time_index,
            station_id, 0x12345678, 1.0, status);

    /* Generate weights errors. */
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE)
            oskar_evaluate_element_weights_errors_d(num_elements,
                    oskar_mem_double_const(gain, status),
                    oskar_mem_double_const(gain_error, status),
                    oskar_mem_double_const(phase, status),
                    oskar_mem_double_const(phase_error, status),
                    oskar_mem_double2(errors, status));
        else if (type == OSKAR_SINGLE)
            oskar_evaluate_element_weights_errors_f(num_elements,
                    oskar_mem_float_const(gain, status),
                    oskar_mem_float_const(gain_error, status),
                    oskar_mem_float_const(phase, status),
                    oskar_mem_float_const(phase_error, status),
                    oskar_mem_float2(errors, status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_DOUBLE)
            oskar_evaluate_element_weights_errors_cuda_d(num_elements,
                    oskar_mem_double_const(gain, status),
                    oskar_mem_double_const(gain_error, status),
                    oskar_mem_double_const(phase, status),
                    oskar_mem_double_const(phase_error, status),
                    oskar_mem_double2(errors, status));
        else if (type == OSKAR_SINGLE)
            oskar_evaluate_element_weights_errors_cuda_f(num_elements,
                    oskar_mem_float_const(gain, status),
                    oskar_mem_float_const(gain_error, status),
                    oskar_mem_float_const(phase, status),
                    oskar_mem_float_const(phase_error, status),
                    oskar_mem_float2(errors, status));
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
            k = oskar_cl_kernel("evaluate_element_weights_errors_double");
        else if (type == OSKAR_SINGLE)
            k = oskar_cl_kernel("evaluate_element_weights_errors_float");
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
                oskar_mem_cl_buffer_const(gain, status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(gain_error, status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(phase, status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(phase_error, status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer(errors, status));
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
