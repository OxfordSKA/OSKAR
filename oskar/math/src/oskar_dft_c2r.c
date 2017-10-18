/*
 * Copyright (c) 2017, The University of Oxford
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

#include "math/oskar_dft_c2r.h"
#include "math/oskar_dft_c2r_2d_cuda.h"
#include "math/oskar_dft_c2r_2d_omp.h"
#include "math/oskar_dft_c2r_3d_cuda.h"
#include "math/oskar_dft_c2r_3d_omp.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_device_utils.h"

/* Utility functions. */
static int oskar_int_round_to_nearest_multiple(int num_to_round, int multiple)
{
   return (num_to_round + multiple - 1) / multiple * multiple;
}

static int oskar_int_range_clamp(int value, int minimum, int maximum)
{
   if (value < minimum)
       return minimum;
   if (value > maximum)
       return maximum;
   return value;
}

void oskar_dft_c2r(
        int num_in,
        double wavenumber,
        const oskar_Mem* x_in,
        const oskar_Mem* y_in,
        const oskar_Mem* z_in,
        const oskar_Mem* data_in,
        const oskar_Mem* weights_in,
        int num_out,
        const oskar_Mem* x_out,
        const oskar_Mem* y_out,
        const oskar_Mem* z_out,
        oskar_Mem* output,
        int* status)
{
    int location, type, is_dbl, is_3d;
    if (*status) return;

    /* Find out what we have. */
    location = oskar_mem_location(output);
    type = oskar_mem_precision(output);
    is_dbl = type & OSKAR_DOUBLE;
    is_3d = (z_in != NULL && z_out != NULL && oskar_mem_length(z_out) > 0);
    if (!oskar_mem_is_complex(data_in) ||
            oskar_mem_precision(data_in) != type ||
            oskar_mem_is_complex(output) ||
            oskar_mem_is_complex(weights_in) ||
            oskar_mem_is_matrix(weights_in))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Check type and location consistency. */
    if (oskar_mem_location(weights_in) != location ||
            oskar_mem_location(x_in) != location ||
            oskar_mem_location(y_in) != location ||
            oskar_mem_location(x_out) != location ||
            oskar_mem_location(y_out) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (oskar_mem_precision(weights_in) != type ||
            oskar_mem_type(x_in) != type ||
            oskar_mem_type(y_in) != type ||
            oskar_mem_type(x_out) != type ||
            oskar_mem_type(y_out) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (is_3d)
    {
        if (oskar_mem_location(z_in) != location ||
                oskar_mem_location(z_out) != location)
        {
            *status = OSKAR_ERR_LOCATION_MISMATCH;
            return;
        }
        if (oskar_mem_type(z_in) != type || oskar_mem_type(z_out) != type)
        {
            *status = OSKAR_ERR_TYPE_MISMATCH;
            return;
        }
    }

    /* Resize output array if needed. */
    if ((int)oskar_mem_length(output) < num_out)
        oskar_mem_realloc(output, (size_t) num_out, status);
    if (*status) return;

    /* Switch on location. */
    if (location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        cl_device_type dev_type;
        cl_kernel k;
        int is_gpu, out_size, max_out_size, start;

        /* Get the appropriate kernel. */
        clGetDeviceInfo(oskar_cl_device_id(),
                CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL);
        is_gpu = dev_type & CL_DEVICE_TYPE_GPU;
        const size_t local_size = is_gpu ? 256 : 128;
        const size_t element_size = is_dbl ?
                sizeof(cl_double) : sizeof(cl_float);
        if (is_3d)
        {
            k = is_gpu ? (is_dbl ?
                    oskar_cl_kernel("dft_c2r_3d_double") :
                    oskar_cl_kernel("dft_c2r_3d_float")) : (is_dbl ?
                            oskar_cl_kernel("dft_c2r_3d_cpu_double") :
                            oskar_cl_kernel("dft_c2r_3d_cpu_float"));
        }
        else
        {
            k = is_gpu ? (is_dbl ?
                    oskar_cl_kernel("dft_c2r_2d_double") :
                    oskar_cl_kernel("dft_c2r_2d_float")) : (is_dbl ?
                            oskar_cl_kernel("dft_c2r_2d_cpu_double") :
                            oskar_cl_kernel("dft_c2r_2d_cpu_float"));
        }
        if (!k)
        {
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            return;
        }

        /* Compute the maximum manageable output chunk size. */
        /* Product of max output and input sizes. */
        max_out_size = 8192 * (is_dbl ? 32768 : 65536);
        max_out_size /= num_in;
        max_out_size = oskar_int_round_to_nearest_multiple(max_out_size, 1024);
        max_out_size = oskar_int_range_clamp(max_out_size,
                (int) local_size * 2,
                (int) local_size * (is_dbl ? 80 : 160));

        /* Loop over output chunks. */
        for (start = 0; start < num_out; start += max_out_size)
        {
            cl_event event;
            cl_int error, n_in, n_out;
            cl_uint arg = 0;
            oskar_Mem *x_chunk, *y_chunk, *z_chunk, *o_chunk;
            if (*status) break;

            /* Get the chunk size. */
            out_size = num_out - start;
            if (out_size > max_out_size) out_size = max_out_size;

            /* Create sub-buffers for the chunk. */
            x_chunk = oskar_mem_create_alias(x_out, start, out_size, status);
            y_chunk = oskar_mem_create_alias(y_out, start, out_size, status);
            z_chunk = oskar_mem_create_alias(z_out, start, out_size, status);
            o_chunk = oskar_mem_create_alias(output, start, out_size, status);

            /* Set kernel arguments. */
            n_in = (cl_int) num_in;
            n_out = (cl_int) out_size;
            error = clSetKernelArg(k, arg++, sizeof(cl_int), &n_in);
            if (is_dbl)
            {
                cl_double w = (cl_double) wavenumber;
                error |= clSetKernelArg(k, arg++, sizeof(cl_double), &w);
            }
            else
            {
                cl_float w = (cl_float) wavenumber;
                error |= clSetKernelArg(k, arg++, sizeof(cl_float), &w);
            }
            error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                    oskar_mem_cl_buffer_const(x_in, status));
            error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                    oskar_mem_cl_buffer_const(y_in, status));
            if (is_3d)
                error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                        oskar_mem_cl_buffer_const(z_in, status));
            error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                    oskar_mem_cl_buffer_const(data_in, status));
            error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                    oskar_mem_cl_buffer_const(weights_in, status));
            error |= clSetKernelArg(k, arg++, sizeof(cl_int), &n_out);
            error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                    oskar_mem_cl_buffer_const(x_chunk, status));
            error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                    oskar_mem_cl_buffer_const(y_chunk, status));
            if (is_3d)
                error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                        oskar_mem_cl_buffer_const(z_chunk, status));
            error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                    oskar_mem_cl_buffer(o_chunk, status));
            if (is_gpu)
            {
                /* max_in_chunk must be multiple of 16. */
                const cl_int max_in_chunk = is_3d ?
                        (is_dbl ? 384 : 800) : (is_dbl ? 448 : 896);
                const size_t local_mem_size = max_in_chunk * element_size;
                error |= clSetKernelArg(k, arg++, sizeof(cl_int),
                        &max_in_chunk);
                error |= clSetKernelArg(k, arg++, 2 * local_mem_size, 0);
                error |= clSetKernelArg(k, arg++, 2 * local_mem_size, 0);
                if (is_3d)
                    error |= clSetKernelArg(k, arg++, local_mem_size, 0);
            }
            if (!*status && error != CL_SUCCESS)
                *status = OSKAR_ERR_INVALID_ARGUMENT;
            if (!*status)
            {
                /* Launch kernel on current command queue. */
                const size_t global_size = ((out_size + local_size - 1) /
                        local_size) * local_size;
                error = clEnqueueNDRangeKernel(oskar_cl_command_queue(), k, 1,
                        NULL, &global_size, &local_size, 0, NULL, &event);
                if (error != CL_SUCCESS)
                    *status = OSKAR_ERR_KERNEL_LAUNCH_FAILURE;
            }

            /* Free sub-buffers. */
            oskar_mem_free(x_chunk, status);
            oskar_mem_free(y_chunk, status);
            oskar_mem_free(z_chunk, status);
            oskar_mem_free(o_chunk, status);
        }
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (is_3d)
        {
            if (is_dbl)
                oskar_dft_c2r_3d_cuda_d(num_in, wavenumber,
                        oskar_mem_double_const(x_in, status),
                        oskar_mem_double_const(y_in, status),
                        oskar_mem_double_const(z_in, status),
                        oskar_mem_double2_const(data_in, status),
                        oskar_mem_double_const(weights_in, status),
                        num_out, oskar_mem_double_const(x_out, status),
                        oskar_mem_double_const(y_out, status),
                        oskar_mem_double_const(z_out, status),
                        oskar_mem_double(output, status));
            else
                oskar_dft_c2r_3d_cuda_f(num_in, (float)wavenumber,
                        oskar_mem_float_const(x_in, status),
                        oskar_mem_float_const(y_in, status),
                        oskar_mem_float_const(z_in, status),
                        oskar_mem_float2_const(data_in, status),
                        oskar_mem_float_const(weights_in, status),
                        num_out, oskar_mem_float_const(x_out, status),
                        oskar_mem_float_const(y_out, status),
                        oskar_mem_float_const(z_out, status),
                        oskar_mem_float(output, status));
        }
        else
        {
            if (is_dbl)
                oskar_dft_c2r_2d_cuda_d(num_in, wavenumber,
                        oskar_mem_double_const(x_in, status),
                        oskar_mem_double_const(y_in, status),
                        oskar_mem_double2_const(data_in, status),
                        oskar_mem_double_const(weights_in, status),
                        num_out, oskar_mem_double_const(x_out, status),
                        oskar_mem_double_const(y_out, status),
                        oskar_mem_double(output, status));
            else
                oskar_dft_c2r_2d_cuda_f(num_in, (float)wavenumber,
                        oskar_mem_float_const(x_in, status),
                        oskar_mem_float_const(y_in, status),
                        oskar_mem_float2_const(data_in, status),
                        oskar_mem_float_const(weights_in, status),
                        num_out, oskar_mem_float_const(x_out, status),
                        oskar_mem_float_const(y_out, status),
                        oskar_mem_float(output, status));
        }
        oskar_device_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_CPU)
    {
        if (is_3d)
        {
            if (is_dbl)
                oskar_dft_c2r_3d_omp_d(num_in, wavenumber,
                        oskar_mem_double_const(x_in, status),
                        oskar_mem_double_const(y_in, status),
                        oskar_mem_double_const(z_in, status),
                        oskar_mem_double2_const(data_in, status),
                        oskar_mem_double_const(weights_in, status),
                        num_out, oskar_mem_double_const(x_out, status),
                        oskar_mem_double_const(y_out, status),
                        oskar_mem_double_const(z_out, status),
                        oskar_mem_double(output, status));
            else
                oskar_dft_c2r_3d_omp_f(num_in, (float)wavenumber,
                        oskar_mem_float_const(x_in, status),
                        oskar_mem_float_const(y_in, status),
                        oskar_mem_float_const(z_in, status),
                        oskar_mem_float2_const(data_in, status),
                        oskar_mem_float_const(weights_in, status),
                        num_out, oskar_mem_float_const(x_out, status),
                        oskar_mem_float_const(y_out, status),
                        oskar_mem_float_const(z_out, status),
                        oskar_mem_float(output, status));
        }
        else
        {
            if (is_dbl)
                oskar_dft_c2r_2d_omp_d(num_in, wavenumber,
                        oskar_mem_double_const(x_in, status),
                        oskar_mem_double_const(y_in, status),
                        oskar_mem_double2_const(data_in, status),
                        oskar_mem_double_const(weights_in, status),
                        num_out, oskar_mem_double_const(x_out, status),
                        oskar_mem_double_const(y_out, status),
                        oskar_mem_double(output, status));
            else
                oskar_dft_c2r_2d_omp_f(num_in, (float)wavenumber,
                        oskar_mem_float_const(x_in, status),
                        oskar_mem_float_const(y_in, status),
                        oskar_mem_float2_const(data_in, status),
                        oskar_mem_float_const(weights_in, status),
                        num_out, oskar_mem_float_const(x_out, status),
                        oskar_mem_float_const(y_out, status),
                        oskar_mem_float(output, status));
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}
