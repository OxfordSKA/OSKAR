/*
 * Copyright (c) 2013-2017, The University of Oxford
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

#include "math/oskar_gaussian_circular.h"
#include "math/oskar_gaussian_circular_cuda.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_device_utils.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_gaussian_circular_complex_f(int n, const float* x,
        const float* y, float std, float2* z)
{
    int i;
    float inv_2_var;
    inv_2_var = 1.0f / (2.0f * std * std);

    for (i = 0; i < n; ++i)
    {
        float x_, y_, arg;
        x_ = x[i];
        y_ = y[i];
        arg = (x_*x_ + y_*y_) * inv_2_var;
        z[i].x = expf(-arg);
        z[i].y = 0.0f;
    }
}

void oskar_gaussian_circular_matrix_f(int n, const float* x,
        const float* y, float std, float4c* z)
{
    int i;
    float inv_2_var;
    inv_2_var = 1.0f / (2.0f * std * std);

    for (i = 0; i < n; ++i)
    {
        float x_, y_, arg, value;
        x_ = x[i];
        y_ = y[i];
        arg = (x_*x_ + y_*y_) * inv_2_var;
        value = expf(-arg);
        z[i].a.x = value;
        z[i].a.y = 0.0f;
        z[i].b.x = 0.0f;
        z[i].b.y = 0.0f;
        z[i].c.x = 0.0f;
        z[i].c.y = 0.0f;
        z[i].d.x = value;
        z[i].d.y = 0.0f;
    }
}


/* Double precision. */
void oskar_gaussian_circular_complex_d(int n, const double* x,
        const double* y, double std, double2* z)
{
    int i;
    double inv_2_var;
    inv_2_var = 1.0 / (2.0 * std * std);

    for (i = 0; i < n; ++i)
    {
        double x_, y_, arg;
        x_ = x[i];
        y_ = y[i];
        arg = (x_*x_ + y_*y_) * inv_2_var;
        z[i].x = exp(-arg);
        z[i].y = 0.0;
    }
}

void oskar_gaussian_circular_matrix_d(int n, const double* x,
        const double* y, double std, double4c* z)
{
    int i;
    double inv_2_var;
    inv_2_var = 1.0 / (2.0 * std * std);

    for (i = 0; i < n; ++i)
    {
        double x_, y_, arg, value;
        x_ = x[i];
        y_ = y[i];
        arg = (x_*x_ + y_*y_) * inv_2_var;
        value = exp(-arg);
        z[i].a.x = value;
        z[i].a.y = 0.0;
        z[i].b.x = 0.0;
        z[i].b.y = 0.0;
        z[i].c.x = 0.0;
        z[i].c.y = 0.0;
        z[i].d.x = value;
        z[i].d.y = 0.0;
    }
}

void oskar_gaussian_circular(int num_points,
        const oskar_Mem* l, const oskar_Mem* m, double std,
        oskar_Mem* out, int* status)
{
    int is_scalar, type, location;
#ifdef OSKAR_HAVE_OPENCL
    cl_kernel k = 0;
#endif

    /* Get type and check consistency. */
    if (*status) return;
    type = oskar_mem_precision(out);
    is_scalar = oskar_mem_is_scalar(out);
    if (type != oskar_mem_type(l) || type != oskar_mem_type(m))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (!oskar_mem_is_complex(out))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Get location and check consistency. */
    location = oskar_mem_location(out);
    if (location != oskar_mem_location(l) ||
            location != oskar_mem_location(m))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check that length of input arrays are consistent. */
    if ((int)oskar_mem_length(l) < num_points ||
            (int)oskar_mem_length(m) < num_points)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Resize output array if needed. */
    if ((int)oskar_mem_length(out) < num_points)
        oskar_mem_realloc(out, num_points, status);
    if (*status) return;

    if (type == OSKAR_DOUBLE)
    {
        const double *l_, *m_;
        l_ = oskar_mem_double_const(l, status);
        m_ = oskar_mem_double_const(m, status);

        if (location == OSKAR_CPU)
        {
            if (is_scalar)
                oskar_gaussian_circular_complex_d(num_points, l_, m_, std,
                        oskar_mem_double2(out, status));
            else
                oskar_gaussian_circular_matrix_d(num_points, l_, m_, std,
                        oskar_mem_double4c(out, status));
        }
        else if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            if (is_scalar)
                oskar_gaussian_circular_cuda_complex_d(num_points, l_, m_, std,
                        oskar_mem_double2(out, status));
            else
                oskar_gaussian_circular_cuda_matrix_d(num_points, l_, m_, std,
                        oskar_mem_double4c(out, status));
            oskar_device_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location & OSKAR_CL)
        {
#ifdef OSKAR_HAVE_OPENCL
            k = is_scalar ?
                    oskar_cl_kernel("gaussian_circular_complex_double") :
                    oskar_cl_kernel("gaussian_circular_matrix_double");
#else
            *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
    }
    else if (type == OSKAR_SINGLE)
    {
        const float *l_, *m_;
        l_ = oskar_mem_float_const(l, status);
        m_ = oskar_mem_float_const(m, status);

        if (location == OSKAR_CPU)
        {
            if (is_scalar)
                oskar_gaussian_circular_complex_f(num_points, l_, m_,
                        (float)std, oskar_mem_float2(out, status));
            else
                oskar_gaussian_circular_matrix_f(num_points, l_, m_,
                        (float)std, oskar_mem_float4c(out, status));
        }
        else if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            if (is_scalar)
                oskar_gaussian_circular_cuda_complex_f(num_points, l_, m_,
                        (float)std, oskar_mem_float2(out, status));
            else
                oskar_gaussian_circular_cuda_matrix_f(num_points, l_, m_,
                        (float)std, oskar_mem_float4c(out, status));
            oskar_device_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location & OSKAR_CL)
        {
#ifdef OSKAR_HAVE_OPENCL
            k = is_scalar ?
                    oskar_cl_kernel("gaussian_circular_complex_float") :
                    oskar_cl_kernel("gaussian_circular_matrix_float");
#else
            *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
#endif
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
    }
    else
        *status = OSKAR_ERR_BAD_DATA_TYPE;

#ifdef OSKAR_HAVE_OPENCL
    /* Call OpenCL kernel if required. */
    if ((location & OSKAR_CL) && !*status)
    {
        if (k)
        {
            cl_device_type dev_type;
            cl_int error, gpu, n;
            size_t global_size, local_size;
            double inv_2_var;

            /* Set kernel arguments. */
            clGetDeviceInfo(oskar_cl_device_id(),
                    CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL);
            gpu = dev_type & CL_DEVICE_TYPE_GPU;
            n = (cl_int) num_points;
            error = clSetKernelArg(k, 0, sizeof(cl_int), &n);
            error |= clSetKernelArg(k, 1, sizeof(cl_mem),
                    oskar_mem_cl_buffer_const(l, status));
            error |= clSetKernelArg(k, 2, sizeof(cl_mem),
                    oskar_mem_cl_buffer_const(m, status));
            inv_2_var = 1.0 / (2.0 * std * std);
            if (type == OSKAR_SINGLE)
            {
                cl_float w = (cl_float) inv_2_var;
                error |= clSetKernelArg(k, 3, sizeof(cl_float), &w);
            }
            else
            {
                cl_double w = (cl_double) inv_2_var;
                error |= clSetKernelArg(k, 3, sizeof(cl_double), &w);
            }
            error |= clSetKernelArg(k, 4, sizeof(cl_mem),
                    oskar_mem_cl_buffer(out, status));
            if (*status) return;
            if (error != CL_SUCCESS)
            {
                *status = OSKAR_ERR_INVALID_ARGUMENT;
                return;
            }

            /* Launch kernel on current command queue. */
            local_size = gpu ? 256 : 128;
            global_size = ((num_points + local_size - 1) / local_size) *
                    local_size;
            error = clEnqueueNDRangeKernel(oskar_cl_command_queue(), k, 1, NULL,
                        &global_size, &local_size, 0, NULL, NULL);
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
    }
#endif
}

#ifdef __cplusplus
}
#endif
