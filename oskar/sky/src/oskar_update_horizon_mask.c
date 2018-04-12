/*
 * Copyright (c) 2013-2018, The University of Oxford
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

#include "sky/oskar_update_horizon_mask.h"
#include "sky/oskar_update_horizon_mask_cuda.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_device_utils.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_update_horizon_mask(int num_sources, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, const double ha0_rad,
        const double dec0_rad, const double lat_rad, oskar_Mem* mask,
        int* status)
{
    int i, type, location, *mask_;
    double cos_ha0, sin_dec0, cos_dec0, sin_lat, cos_lat;
    double ll, mm, nn;
    if (*status) return;
    type = oskar_mem_precision(l);
    location = oskar_mem_location(mask);
    mask_ = oskar_mem_int(mask, status);
    cos_ha0  = cos(ha0_rad);
    sin_dec0 = sin(dec0_rad);
    cos_dec0 = cos(dec0_rad);
    sin_lat  = sin(lat_rad);
    cos_lat  = cos(lat_rad);
    ll = cos_lat * sin(ha0_rad);
    mm = sin_lat * cos_dec0 - cos_lat * cos_ha0 * sin_dec0;
    nn = sin_lat * sin_dec0 + cos_lat * cos_ha0 * cos_dec0;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            float ll_, mm_, nn_;
            const float *l_, *m_, *n_;
            l_ = oskar_mem_float_const(l, status);
            m_ = oskar_mem_float_const(m, status);
            n_ = oskar_mem_float_const(n, status);
            ll_ = (float) ll;
            mm_ = (float) mm;
            nn_ = (float) nn;
            for (i = 0; i < num_sources; ++i)
                mask_[i] |= ((l_[i] * ll_ + m_[i] * mm_ + n_[i] * nn_) > 0.f);
        }
        else if (type == OSKAR_DOUBLE)
        {
            const double *l_, *m_, *n_;
            l_ = oskar_mem_double_const(l, status);
            m_ = oskar_mem_double_const(m, status);
            n_ = oskar_mem_double_const(n, status);
            for (i = 0; i < num_sources; ++i)
                mask_[i] |= ((l_[i] * ll + m_[i] * mm + n_[i] * nn) > 0.);
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
            oskar_update_horizon_mask_cuda_f(num_sources,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status),
                    oskar_mem_float_const(n, status),
                    (float) ll, (float) mm, (float) nn, mask_);
        else if (type == OSKAR_DOUBLE)
            oskar_update_horizon_mask_cuda_d(num_sources,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status),
                    oskar_mem_double_const(n, status),
                    ll, mm, nn, mask_);
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
            k = oskar_cl_kernel("update_horizon_mask_double");
        else if (type == OSKAR_SINGLE)
            k = oskar_cl_kernel("update_horizon_mask_float");
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
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(l, status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(m, status));
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer_const(n, status));
        if (type == OSKAR_SINGLE)
        {
            const cl_float ll_ = (cl_float) ll;
            const cl_float mm_ = (cl_float) mm;
            const cl_float nn_ = (cl_float) nn;
            error |= clSetKernelArg(k, arg++, sizeof(cl_float), &ll_);
            error |= clSetKernelArg(k, arg++, sizeof(cl_float), &mm_);
            error |= clSetKernelArg(k, arg++, sizeof(cl_float), &nn_);
        }
        else if (type == OSKAR_DOUBLE)
        {
            const cl_double ll_ = (cl_double) ll;
            const cl_double mm_ = (cl_double) mm;
            const cl_double nn_ = (cl_double) nn;
            error |= clSetKernelArg(k, arg++, sizeof(cl_double), &ll_);
            error |= clSetKernelArg(k, arg++, sizeof(cl_double), &mm_);
            error |= clSetKernelArg(k, arg++, sizeof(cl_double), &nn_);
        }
        error |= clSetKernelArg(k, arg++, sizeof(cl_mem),
                oskar_mem_cl_buffer(mask, status));
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
