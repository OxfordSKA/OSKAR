/*
 * Copyright (c) 2019, The University of Oxford
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

#include "math/define_legendre_polynomial.h"
#include "math/define_spherical_harmonic.h"
#include "math/private_spherical_harmonic.h"
#include "math/oskar_spherical_harmonic_sum.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_SPHERICAL_HARMONIC_SUM_REAL(spherical_harmonic_sum_real_float, float)
OSKAR_SPHERICAL_HARMONIC_SUM_REAL(spherical_harmonic_sum_real_double, double)

void oskar_spherical_harmonic_sum(const oskar_SphericalHarmonic* h,
        int num_points, const oskar_Mem* theta, const oskar_Mem* phi,
        int stride, int offset_out, oskar_Mem* surface, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(h->coeff);
    const int location = oskar_mem_location(surface);
    const int l_max = h->l_max;
    const oskar_Mem* coeff = h->coeff;
    if (oskar_mem_length(coeff) < (size_t) ((l_max + 1) * (l_max + 1)))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        switch (type)
        {
        case OSKAR_SINGLE:
            spherical_harmonic_sum_real_float(num_points,
                    oskar_mem_float_const(theta, status),
                    oskar_mem_float_const(phi, status), l_max,
                    oskar_mem_float_const(coeff, status), stride, offset_out,
                    oskar_mem_float(surface, status));
            break;
        case OSKAR_SINGLE_COMPLEX:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        case OSKAR_DOUBLE:
            spherical_harmonic_sum_real_double(num_points,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status), l_max,
                    oskar_mem_double_const(coeff, status), stride, offset_out,
                    oskar_mem_double(surface, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        switch (type)
        {
        case OSKAR_SINGLE:
            k = "spherical_harmonic_sum_real_float";
            break;
        case OSKAR_SINGLE_COMPLEX:
            k = "spherical_harmonic_sum_complex_float";
            break;
        case OSKAR_DOUBLE:
            k = "spherical_harmonic_sum_real_double";
            break;
        case OSKAR_DOUBLE_COMPLEX:
            k = "spherical_harmonic_sum_complex_double";
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]);
        const oskar_Arg arg[] = {
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(theta)},
                {PTR_SZ, oskar_mem_buffer_const(phi)},
                {INT_SZ, &l_max},
                {PTR_SZ, oskar_mem_buffer_const(coeff)},
                {INT_SZ, &stride},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer(surface)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(arg) / sizeof(oskar_Arg), arg, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
