/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/define_multiply.h"
#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "mem/define_mem_multiply.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
OSKAR_MEM_MUL_RR_R( M_CAT(mem_mul_rr_r_, float), float)
OSKAR_MEM_MUL_CC_C( M_CAT(mem_mul_cc_c_, float), float2)
OSKAR_MEM_MUL_CC_M( M_CAT(mem_mul_cc_m_, float), float, float2, float4c)
OSKAR_MEM_MUL_CM_M( M_CAT(mem_mul_cm_m_, float), float2, float4c)
OSKAR_MEM_MUL_MC_M( M_CAT(mem_mul_mc_m_, float), float2, float4c)
OSKAR_MEM_MUL_MM_M( M_CAT(mem_mul_mm_m_, float), float2, float4c)

/* Double precision. */
OSKAR_MEM_MUL_RR_R( M_CAT(mem_mul_rr_r_, double), double)
OSKAR_MEM_MUL_CC_C( M_CAT(mem_mul_cc_c_, double), double2)
OSKAR_MEM_MUL_CC_M( M_CAT(mem_mul_cc_m_, double), double, double2, double4c)
OSKAR_MEM_MUL_CM_M( M_CAT(mem_mul_cm_m_, double), double2, double4c)
OSKAR_MEM_MUL_MC_M( M_CAT(mem_mul_mc_m_, double), double2, double4c)
OSKAR_MEM_MUL_MM_M( M_CAT(mem_mul_mm_m_, double), double2, double4c)

void oskar_mem_multiply(
        oskar_Mem* out,
        const oskar_Mem* in1,
        const oskar_Mem* in2,
        size_t offset_out,
        size_t offset_in1,
        size_t offset_in2,
        size_t num_elements,
        int* status)
{
    oskar_Mem *a_temp = 0, *b_temp = 0;
    const oskar_Mem *a_ = 0, *b_ = 0; /* Pointers. */
    if (num_elements == 0) return;
    const int location = oskar_mem_location(out);
    const unsigned int off_a = (unsigned int) offset_in1;
    const unsigned int off_b = (unsigned int) offset_in2;
    const unsigned int off_c = (unsigned int) offset_out;
    const unsigned int n = (unsigned int) num_elements;
    if (*status) return;
    a_ = in1;
    b_ = in2;
    if (oskar_mem_location(in1) != location)
    {
        a_temp = oskar_mem_create_copy(in1, location, status);
        a_ = a_temp;
    }
    if (oskar_mem_location(in2) != location)
    {
        b_temp = oskar_mem_create_copy(in2, location, status);
        b_ = b_temp;
    }
    if (location == OSKAR_CPU)
    {
        void *c = out->data;
        const void *a = a_->data, *b = b_->data;
        /* Check if types are all the same. */
        if (out->type == in1->type && out->type == in2->type)
        {
            switch (out->type)
            {
            case OSKAR_DOUBLE:
                mem_mul_rr_r_double(off_a, off_b, off_c, n,
                        (const double*)a, (const double*)b, (double*)c);
                break;
            case OSKAR_DOUBLE_COMPLEX:
                mem_mul_cc_c_double(off_a, off_b, off_c, n,
                        (const double2*)a, (const double2*)b, (double2*)c);
                break;
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
                mem_mul_mm_m_double(off_a, off_b, off_c, n,
                        (const double4c*)a, (const double4c*)b, (double4c*)c);
                break;
            case OSKAR_SINGLE:
                mem_mul_rr_r_float(off_a, off_b, off_c, n,
                        (const float*)a, (const float*)b, (float*)c);
                break;
            case OSKAR_SINGLE_COMPLEX:
                mem_mul_cc_c_float(off_a, off_b, off_c, n,
                        (const float2*)a, (const float2*)b, (float2*)c);
                break;
            case OSKAR_SINGLE_COMPLEX_MATRIX:
                mem_mul_mm_m_float(off_a, off_b, off_c, n,
                        (const float4c*)a, (const float4c*)b, (float4c*)c);
                break;
            default:
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                break;
            }
        }
        else
        {
            switch (out->type)
            {
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
            {
                switch (in1->type)
                {
                case OSKAR_DOUBLE_COMPLEX:
                    if (in2->type == in1->type)
                    {
                        mem_mul_cc_m_double(off_a, off_b, off_c, n,
                                (const double2*)a, (const double2*)b,
                                (double4c*)c);
                    }
                    else if (in2->type == out->type)
                    {
                        mem_mul_cm_m_double(off_a, off_b, off_c, n,
                                (const double2*)a, (const double4c*)b,
                                (double4c*)c);
                    }
                    else
                    {
                        *status = OSKAR_ERR_TYPE_MISMATCH;
                    }
                    break;
                case OSKAR_DOUBLE_COMPLEX_MATRIX:
                    if (in2->type == OSKAR_DOUBLE_COMPLEX)
                    {
                        mem_mul_mc_m_double(off_a, off_b, off_c, n,
                                (const double4c*)a, (const double2*)b,
                                (double4c*)c);
                    }
                    else
                    {
                        *status = OSKAR_ERR_TYPE_MISMATCH;
                    }
                    break;
                default:
                    *status = OSKAR_ERR_TYPE_MISMATCH;
                    break;
                }
                break;
            }
            case OSKAR_SINGLE_COMPLEX_MATRIX:
            {
                switch (in1->type)
                {
                case OSKAR_SINGLE_COMPLEX:
                    if (in2->type == in1->type)
                    {
                        mem_mul_cc_m_float(off_a, off_b, off_c, n,
                                (const float2*)a, (const float2*)b,
                                (float4c*)c);
                    }
                    else if (in2->type == out->type)
                    {
                        mem_mul_cm_m_float(off_a, off_b, off_c, n,
                                (const float2*)a, (const float4c*)b,
                                (float4c*)c);
                    }
                    else
                    {
                        *status = OSKAR_ERR_TYPE_MISMATCH;
                    }
                    break;
                case OSKAR_SINGLE_COMPLEX_MATRIX:
                    if (in2->type == OSKAR_SINGLE_COMPLEX)
                    {
                        mem_mul_mc_m_float(off_a, off_b, off_c, n,
                                (const float4c*)a, (const float2*)b,
                                (float4c*)c);
                    }
                    else
                    {
                        *status = OSKAR_ERR_TYPE_MISMATCH;
                    }
                    break;
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
        }
    }
    else
    {
        const char* k = 0;
        /* Check if types are all the same. */
        if (out->type == in1->type && out->type == in2->type)
        {
            switch (out->type)
            {
            case OSKAR_DOUBLE:                k = "mem_mul_rr_r_double"; break;
            case OSKAR_DOUBLE_COMPLEX:        k = "mem_mul_cc_c_double"; break;
            case OSKAR_DOUBLE_COMPLEX_MATRIX: k = "mem_mul_mm_m_double"; break;
            case OSKAR_SINGLE:                k = "mem_mul_rr_r_float"; break;
            case OSKAR_SINGLE_COMPLEX:        k = "mem_mul_cc_c_float"; break;
            case OSKAR_SINGLE_COMPLEX_MATRIX: k = "mem_mul_mm_m_float"; break;
            default:
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                break;
            }
        }
        else
        {
            switch (out->type)
            {
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
            {
                switch (in1->type)
                {
                case OSKAR_DOUBLE_COMPLEX:
                    if (in2->type == in1->type)
                    {
                        k = "mem_mul_cc_m_double";
                    }
                    else if (in2->type == out->type)
                    {
                        k = "mem_mul_cm_m_double";
                    }
                    else
                    {
                        *status = OSKAR_ERR_TYPE_MISMATCH;
                    }
                    break;
                case OSKAR_DOUBLE_COMPLEX_MATRIX:
                    if (in2->type == OSKAR_DOUBLE_COMPLEX)
                    {
                        k = "mem_mul_mc_m_double";
                    }
                    else
                    {
                        *status = OSKAR_ERR_TYPE_MISMATCH;
                    }
                    break;
                default:
                    *status = OSKAR_ERR_TYPE_MISMATCH;
                    break;
                }
                break;
            }
            case OSKAR_SINGLE_COMPLEX_MATRIX:
            {
                switch (in1->type)
                {
                case OSKAR_SINGLE_COMPLEX:
                    if (in2->type == in1->type)
                    {
                        k = "mem_mul_cc_m_float";
                    }
                    else if (in2->type == out->type)
                    {
                        k = "mem_mul_cm_m_float";
                    }
                    else
                    {
                        *status = OSKAR_ERR_TYPE_MISMATCH;
                    }
                    break;
                case OSKAR_SINGLE_COMPLEX_MATRIX:
                    if (in2->type == OSKAR_SINGLE_COMPLEX)
                    {
                        k = "mem_mul_mc_m_float";
                    }
                    else
                    {
                        *status = OSKAR_ERR_TYPE_MISMATCH;
                    }
                    break;
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
        }
        if (!*status)
        {
            size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
            oskar_device_check_local_size(location, 0, local_size);
            global_size[0] = oskar_device_global_size(
                    num_elements, local_size[0]);
            const oskar_Arg args[] = {
                    {INT_SZ, &off_a},
                    {INT_SZ, &off_b},
                    {INT_SZ, &off_c},
                    {INT_SZ, &n},
                    {PTR_SZ, oskar_mem_buffer_const(a_)},
                    {PTR_SZ, oskar_mem_buffer_const(b_)},
                    {PTR_SZ, oskar_mem_buffer(out)}
            };
            oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                    sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
        }
    }

    /* Free temporary arrays. */
    oskar_mem_free(a_temp, status);
    oskar_mem_free(b_temp, status);
}

#ifdef __cplusplus
}
#endif
