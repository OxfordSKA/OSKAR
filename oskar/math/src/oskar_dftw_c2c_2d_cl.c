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

#include "math/oskar_dftw_c2c_2d_cl.h"
#include "utility/oskar_cl_utils.h"

void oskar_dftw_c2c_2d_cl(unsigned int num_in, double wavenumber,
        const oskar_Mem* x_in, const oskar_Mem* y_in,
        const oskar_Mem* weights_in, unsigned int num_out,
        const oskar_Mem* x_out, const oskar_Mem* y_out,
        const oskar_Mem* data, oskar_Mem* output, int* status)
{
    cl_device_type dev_type;
    cl_event event;
    cl_int dbl, gpu, error, n_in, n_out;
    cl_kernel k;
    size_t global_size, local_size;
    if (*status) return;

    /* Get the appropriate kernel. */
    clGetDeviceInfo(oskar_cl_device_id(),
            CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL);
    gpu = dev_type & CL_DEVICE_TYPE_GPU;
    dbl = oskar_mem_precision(output) & OSKAR_DOUBLE;
    k = gpu ? (dbl ?
            oskar_cl_kernel("dftw_c2c_2d_double") :
            oskar_cl_kernel("dftw_c2c_2d_float")) : (dbl ?
                    oskar_cl_kernel("dftw_c2c_2d_cpu_double") :
                    oskar_cl_kernel("dftw_c2c_2d_cpu_float"));
    if (!k)
    {
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        return;
    }

    /* Set kernel arguments. */
    n_in = (cl_int) num_in;
    n_out = (cl_int) num_out;
    error = clSetKernelArg(k, 0, sizeof(cl_int), &n_in);
    if (dbl)
    {
        cl_double w = (cl_double) wavenumber;
        error |= clSetKernelArg(k, 1, sizeof(cl_double), &w);
    }
    else
    {
        cl_float w = (cl_float) wavenumber;
        error |= clSetKernelArg(k, 1, sizeof(cl_float), &w);
    }
    error |= clSetKernelArg(k, 2, sizeof(cl_mem),
            oskar_mem_cl_buffer_const(x_in, status));
    error |= clSetKernelArg(k, 3, sizeof(cl_mem),
            oskar_mem_cl_buffer_const(y_in, status));
    error |= clSetKernelArg(k, 4, sizeof(cl_mem),
            oskar_mem_cl_buffer_const(weights_in, status));
    error |= clSetKernelArg(k, 5, sizeof(cl_int), &n_out);
    error |= clSetKernelArg(k, 6, sizeof(cl_mem),
            oskar_mem_cl_buffer_const(x_out, status));
    error |= clSetKernelArg(k, 7, sizeof(cl_mem),
            oskar_mem_cl_buffer_const(y_out, status));
    error |= clSetKernelArg(k, 8, sizeof(cl_mem),
            oskar_mem_cl_buffer_const(data, status));
    error |= clSetKernelArg(k, 9, sizeof(cl_mem),
            oskar_mem_cl_buffer(output, status));
    if (gpu)
    {
        const cl_int max_in_chunk = dbl ? 448 : 896; /* Multiple of 16. */
        const size_t local_mem_size = max_in_chunk * (dbl ?
                sizeof(cl_double) : sizeof(cl_float));
        error |= clSetKernelArg(k, 10, sizeof(cl_int), &max_in_chunk);
        error |= clSetKernelArg(k, 11, 2 * local_mem_size, 0);
        error |= clSetKernelArg(k, 12, 2 * local_mem_size, 0);
    }
    if (*status) return;
    if (error != CL_SUCCESS)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Launch kernel on current command queue. */
    local_size = gpu ? 256 : 128;
    global_size = ((num_out + local_size - 1) / local_size) * local_size;
    error = clEnqueueNDRangeKernel(oskar_cl_command_queue(), k, 1, NULL,
                &global_size, &local_size, 0, NULL, &event);
    if (error != CL_SUCCESS)
    {
        *status = OSKAR_ERR_KERNEL_LAUNCH_FAILURE;
        return;
    }
}
