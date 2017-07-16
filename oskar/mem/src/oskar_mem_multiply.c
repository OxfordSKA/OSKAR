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
#include "mem/oskar_mem_multiply_cuda.h"
#include "mem/private_mem.h"

#include "math/oskar_multiply_inline.h"
#include "utility/oskar_cl_utils.h"
#include "utility/oskar_device_utils.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_mem_multiply_rr_r_f(size_t num, float* c,
        const float* a, const float* b)
{
    size_t i;
    for (i = 0; i < num; ++i)
    {
        c[i] = a[i] * b[i];
    }
}

void oskar_mem_multiply_cc_c_f(size_t num, float2* c,
        const float2* a, const float2* b)
{
    size_t i;
    for (i = 0; i < num; ++i)
    {
        float2 ac, bc, cc;
        ac = a[i];
        bc = b[i];
        oskar_multiply_complex_f(&cc, &ac, &bc);
        c[i] = cc;
    }
}

void oskar_mem_multiply_cc_m_f(size_t num, float4c* c,
        const float2* a, const float2* b)
{
    size_t i;
    for (i = 0; i < num; ++i)
    {
        float2 cc;
        float4c m;
        oskar_multiply_complex_f(&cc, &a[i], &b[i]);

        /* Store result in a matrix. */
        m.a = cc;
        m.b.x = 0.0f;
        m.b.y = 0.0f;
        m.c.x = 0.0f;
        m.c.y = 0.0f;
        m.d = cc;
        c[i] = m;
    }
}

void oskar_mem_multiply_cm_m_f(size_t num, float4c* c,
        const float2* a, const float4c* b)
{
    size_t i;
    for (i = 0; i < num; ++i)
    {
        float2 ac;
        float4c bc;
        ac = a[i];
        bc = b[i];
        oskar_multiply_complex_matrix_complex_scalar_in_place_f(&bc, &ac);
        c[i] = bc;
    }
}

void oskar_mem_multiply_mm_m_f(size_t num, float4c* c,
        const float4c* a, const float4c* b)
{
    size_t i;
    for (i = 0; i < num; ++i)
    {
        float4c ac, bc;
        ac = a[i];
        bc = b[i];
        oskar_multiply_complex_matrix_in_place_f(&ac, &bc);
        c[i] = ac;
    }
}


/* Double precision. */
void oskar_mem_multiply_rr_r_d(size_t num, double* c,
        const double* a, const double* b)
{
    size_t i;
    for (i = 0; i < num; ++i)
    {
        c[i] = a[i] * b[i];
    }
}

void oskar_mem_multiply_cc_c_d(size_t num, double2* c,
        const double2* a, const double2* b)
{
    size_t i;
    for (i = 0; i < num; ++i)
    {
        double2 ac, bc, cc;
        ac = a[i];
        bc = b[i];
        oskar_multiply_complex_d(&cc, &ac, &bc);
        c[i] = cc;
    }
}

void oskar_mem_multiply_cc_m_d(size_t num, double4c* c,
        const double2* a, const double2* b)
{
    size_t i;
    for (i = 0; i < num; ++i)
    {
        double2 cc;
        double4c m;
        oskar_multiply_complex_d(&cc, &a[i], &b[i]);

        /* Store result in a matrix. */
        m.a = cc;
        m.b.x = 0.0;
        m.b.y = 0.0;
        m.c.x = 0.0;
        m.c.y = 0.0;
        m.d = cc;
        c[i] = m;
    }
}

void oskar_mem_multiply_cm_m_d(size_t num, double4c* c,
        const double2* a, const double4c* b)
{
    size_t i;
    for (i = 0; i < num; ++i)
    {
        double2 ac;
        double4c bc;
        ac = a[i];
        bc = b[i];
        oskar_multiply_complex_matrix_complex_scalar_in_place_d(&bc, &ac);
        c[i] = bc;
    }
}

void oskar_mem_multiply_mm_m_d(size_t num, double4c* c,
        const double4c* a, const double4c* b)
{
    size_t i;
    for (i = 0; i < num; ++i)
    {
        double4c ac, bc;
        ac = a[i];
        bc = b[i];
        oskar_multiply_complex_matrix_in_place_d(&ac, &bc);
        c[i] = ac;
    }
}

void oskar_mem_multiply(oskar_Mem* c, oskar_Mem* a, const oskar_Mem* b,
        size_t num, int* status)
{
    oskar_Mem *a_temp = 0, *b_temp = 0;
    const oskar_Mem *a_, *b_; /* Pointers. */
    int use_cpu, use_cuda, use_opencl;
#ifdef OSKAR_HAVE_OPENCL
    cl_kernel k = 0;
#endif

    /* Check if safe to proceed. */
    if (*status) return;

    /* Set default pointer values. */
    a_ = a;
    b_ = b;

    /* Set output to 'a' if not set. */
    if (!c) c = a;

    /* Set the number of elements to multiply. */
    if (num == 0) num = a->num_elements;
    if (num == 0) return;

    /* Check that there are enough elements. */
    if (b->num_elements < num || c->num_elements < num)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Create a temporary copy of the input data if required. */
    use_cpu    = (oskar_mem_location(c) == OSKAR_CPU);
    use_cuda   = (oskar_mem_location(c) == OSKAR_GPU);
    use_opencl = (oskar_mem_location(c) & OSKAR_CL);
#ifndef OSKAR_HAVE_CUDA
    if (use_cuda)
    {
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
        return;
    }
#endif
#ifndef OSKAR_HAVE_OPENCL
    if (use_opencl)
    {
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;
        return;
    }
#endif
    if (use_cpu && oskar_mem_location(a) != OSKAR_CPU)
    {
        a_temp = oskar_mem_create_copy(a, OSKAR_CPU, status);
        a_ = a_temp;
    }
    if (use_cpu && oskar_mem_location(b) != OSKAR_CPU)
    {
        b_temp = oskar_mem_create_copy(b, OSKAR_CPU, status);
        b_ = b_temp;
    }
    if (use_cuda && oskar_mem_location(a) != OSKAR_GPU)
    {
        a_temp = oskar_mem_create_copy(a, OSKAR_GPU, status);
        a_ = a_temp;
    }
    if (use_cuda && oskar_mem_location(b) != OSKAR_GPU)
    {
        b_temp = oskar_mem_create_copy(b, OSKAR_GPU, status);
        b_ = b_temp;
    }
    if (use_opencl && !(oskar_mem_location(a) & OSKAR_CL))
    {
        a_temp = oskar_mem_create_copy(a, OSKAR_CL, status);
        a_ = a_temp;
    }
    if (use_opencl && !(oskar_mem_location(b) & OSKAR_CL))
    {
        b_temp = oskar_mem_create_copy(b, OSKAR_CL, status);
        b_ = b_temp;
    }

    /* Multiply each element of the vectors. */
    /* Early return if types are all the same. */
    if (c->type == a_->type && c->type == b_->type)
    {
        switch (c->type)
        {
        case OSKAR_DOUBLE:
#ifdef OSKAR_HAVE_CUDA
            if (use_cuda)
                oskar_mem_multiply_cuda_rr_r_d((int)num, (double*)c->data,
                        (const double*)a_->data, (const double*)b_->data);
            else
#endif
#ifdef OSKAR_HAVE_OPENCL
            if (use_opencl)
                k = oskar_cl_kernel("mem_multiply_rr_r_double");
            else
#endif
                oskar_mem_multiply_rr_r_d(num, (double*)c->data,
                        (const double*)a_->data, (const double*)b_->data);
            break;
        case OSKAR_DOUBLE_COMPLEX:
#ifdef OSKAR_HAVE_CUDA
            if (use_cuda)
                oskar_mem_multiply_cuda_cc_c_d((int)num, (double2*)c->data,
                        (const double2*)a_->data, (const double2*)b_->data);
            else
#endif
#ifdef OSKAR_HAVE_OPENCL
            if (use_opencl)
                k = oskar_cl_kernel("mem_multiply_cc_c_double");
            else
#endif
                oskar_mem_multiply_cc_c_d(num, (double2*)c->data,
                        (const double2*)a_->data, (const double2*)b_->data);
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
#ifdef OSKAR_HAVE_CUDA
            if (use_cuda)
                oskar_mem_multiply_cuda_mm_m_d((int)num, (double4c*)c->data,
                        (const double4c*)a_->data, (const double4c*)b_->data);
            else
#endif
#ifdef OSKAR_HAVE_OPENCL
            if (use_opencl)
                k = oskar_cl_kernel("mem_multiply_mm_m_double");
            else
#endif
                oskar_mem_multiply_mm_m_d(num, (double4c*)c->data,
                        (const double4c*)a_->data, (const double4c*)b_->data);
            break;
        case OSKAR_SINGLE:
#ifdef OSKAR_HAVE_CUDA
            if (use_cuda)
                oskar_mem_multiply_cuda_rr_r_f((int)num, (float*)c->data,
                        (const float*)a_->data, (const float*)b_->data);
            else
#endif
#ifdef OSKAR_HAVE_OPENCL
            if (use_opencl)
                k = oskar_cl_kernel("mem_multiply_rr_r_float");
            else
#endif
                oskar_mem_multiply_rr_r_f(num, (float*)c->data,
                        (const float*)a_->data, (const float*)b_->data);
            break;
        case OSKAR_SINGLE_COMPLEX:
#ifdef OSKAR_HAVE_CUDA
            if (use_cuda)
                oskar_mem_multiply_cuda_cc_c_f((int)num, (float2*)c->data,
                        (const float2*)a_->data, (const float2*)b_->data);
            else
#endif
#ifdef OSKAR_HAVE_OPENCL
            if (use_opencl)
                k = oskar_cl_kernel("mem_multiply_cc_c_float");
            else
#endif
                oskar_mem_multiply_cc_c_f(num, (float2*)c->data,
                        (const float2*)a_->data, (const float2*)b_->data);
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
#ifdef OSKAR_HAVE_CUDA
            if (use_cuda)
                oskar_mem_multiply_cuda_mm_m_f((int)num, (float4c*)c->data,
                        (const float4c*)a_->data, (const float4c*)b_->data);
            else
#endif
#ifdef OSKAR_HAVE_OPENCL
            if (use_opencl)
                k = oskar_cl_kernel("mem_multiply_mm_m_float");
            else
#endif
                oskar_mem_multiply_mm_m_f(num, (float4c*)c->data,
                        (const float4c*)a_->data, (const float4c*)b_->data);
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        }
        goto done;
    }

    /* Check for allowed mixed types. */
    switch (c->type)
    {
    case OSKAR_DOUBLE_COMPLEX_MATRIX:
    {
        double4c* out = (double4c*)c->data;
        switch (a_->type)
        {
        case OSKAR_DOUBLE_COMPLEX:
        {
            const double2* in_a = (const double2*)a_->data;
            if (b_->type == a_->type)
            {
#ifdef OSKAR_HAVE_CUDA
                if (use_cuda)
                    oskar_mem_multiply_cuda_cc_m_d((int)num, out,
                            in_a, (const double2*)b_->data);
                else
#endif
#ifdef OSKAR_HAVE_OPENCL
                if (use_opencl)
                    k = oskar_cl_kernel("mem_multiply_cc_m_double");
                else
#endif
                    oskar_mem_multiply_cc_m_d(num, out,
                            in_a, (const double2*)b_->data);
            }
            else if (b_->type == c->type)
            {
#ifdef OSKAR_HAVE_CUDA
                if (use_cuda)
                    oskar_mem_multiply_cuda_cm_m_d((int)num, out,
                            in_a, (const double4c*)b_->data);
                else
#endif
#ifdef OSKAR_HAVE_OPENCL
                if (use_opencl)
                    k = oskar_cl_kernel("mem_multiply_cm_m_double");
                else
#endif
                    oskar_mem_multiply_cm_m_d(num, out,
                            in_a, (const double4c*)b_->data);
            }
            else
                *status = OSKAR_ERR_TYPE_MISMATCH;
            break;
        }
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
        {
            if (b_->type == OSKAR_DOUBLE_COMPLEX)
            {
                const double2* in_b = (const double2*)b_->data;
#ifdef OSKAR_HAVE_CUDA
                if (use_cuda)
                    oskar_mem_multiply_cuda_cm_m_d((int)num, out,
                            in_b, (const double4c*)a_->data);
                else
#endif
#ifdef OSKAR_HAVE_OPENCL
                if (use_opencl)
                    k = oskar_cl_kernel("mem_multiply_mc_m_double");
                else
#endif
                    oskar_mem_multiply_cm_m_d(num, out,
                            in_b, (const double4c*)a_->data);
            }
            else
                *status = OSKAR_ERR_TYPE_MISMATCH;
            break;
        }
        default:
            *status = OSKAR_ERR_TYPE_MISMATCH;
            break;
        }
        break;
    }
    case OSKAR_SINGLE_COMPLEX_MATRIX:
    {
        float4c* out = (float4c*)c->data;
        switch (a_->type)
        {
        case OSKAR_SINGLE_COMPLEX:
        {
            const float2* in_a = (const float2*)a_->data;
            if (b_->type == a_->type)
            {
#ifdef OSKAR_HAVE_CUDA
                if (use_cuda)
                    oskar_mem_multiply_cuda_cc_m_f((int)num, out,
                            in_a, (const float2*)b_->data);
                else
#endif
#ifdef OSKAR_HAVE_OPENCL
                if (use_opencl)
                    k = oskar_cl_kernel("mem_multiply_cc_m_float");
                else
#endif
                    oskar_mem_multiply_cc_m_f(num, out,
                            in_a, (const float2*)b_->data);
            }
            else if (b_->type == c->type)
            {
#ifdef OSKAR_HAVE_CUDA
                if (use_cuda)
                    oskar_mem_multiply_cuda_cm_m_f((int)num, out,
                            in_a, (const float4c*)b_->data);
                else
#endif
#ifdef OSKAR_HAVE_OPENCL
                if (use_opencl)
                    k = oskar_cl_kernel("mem_multiply_cm_m_float");
                else
#endif
                    oskar_mem_multiply_cm_m_f(num, out,
                            in_a, (const float4c*)b_->data);
            }
            else
                *status = OSKAR_ERR_TYPE_MISMATCH;
            break;
        }
        case OSKAR_SINGLE_COMPLEX_MATRIX:
        {
            if (b_->type == OSKAR_SINGLE_COMPLEX)
            {
                const float2* in_b = (const float2*)b_->data;
#ifdef OSKAR_HAVE_CUDA
                if (use_cuda)
                    oskar_mem_multiply_cuda_cm_m_f((int)num, out,
                            in_b, (const float4c*)a_->data);
                else
#endif
#ifdef OSKAR_HAVE_OPENCL
                if (use_opencl)
                    k = oskar_cl_kernel("mem_multiply_mc_m_float");
                else
#endif
                    oskar_mem_multiply_cm_m_f(num, out,
                            in_b, (const float4c*)a_->data);
            }
            else
                *status = OSKAR_ERR_TYPE_MISMATCH;
            break;
        }
        default:
            *status = OSKAR_ERR_TYPE_MISMATCH;
            break;
        }
        break;
    }
    default:
        *status = OSKAR_ERR_TYPE_MISMATCH;
        break;
    }

done:
#ifdef OSKAR_HAVE_OPENCL
    /* Call OpenCL kernel if required. */
    if (use_opencl && !*status)
    {
        if (k)
        {
            cl_device_type dev_type;
            cl_int error, gpu = 1, n;
            size_t global_size, local_size;

            /* Set kernel arguments. */
            clGetDeviceInfo(oskar_cl_device_id(),
                    CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL);
            gpu = dev_type & CL_DEVICE_TYPE_GPU;
            n = (cl_int) num;
            error = clSetKernelArg(k, 0, sizeof(cl_int), &n);
            error |= clSetKernelArg(k, 1, sizeof(cl_mem),
                    oskar_mem_cl_buffer_const(a_, status));
            error |= clSetKernelArg(k, 2, sizeof(cl_mem),
                    oskar_mem_cl_buffer_const(b_, status));
            error |= clSetKernelArg(k, 3, sizeof(cl_mem),
                    oskar_mem_cl_buffer(c, status));
            if (*status) return;
            if (error != CL_SUCCESS)
            {
                *status = OSKAR_ERR_INVALID_ARGUMENT;
                return;
            }

            /* Launch kernel on current command queue. */
            local_size = gpu ? 256 : 128;
            global_size = ((num + local_size - 1) / local_size) * local_size;
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

    /* Free temporary arrays. */
#ifdef OSKAR_HAVE_CUDA
    if (use_cuda) oskar_device_check_error(status);
#endif
    oskar_mem_free(a_temp, status);
    oskar_mem_free(b_temp, status);
}

#ifdef __cplusplus
}
#endif
