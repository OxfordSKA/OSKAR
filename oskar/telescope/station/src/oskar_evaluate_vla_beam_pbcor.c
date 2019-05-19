/*
 * Copyright (c) 2013-2019, The University of Oxford
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

#include "telescope/station/define_evaluate_vla_beam_pbcor.h"
#include "telescope/station/oskar_evaluate_vla_beam_pbcor.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"
#include "math/oskar_find_closest_match.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

static const double freqs_ghz[] = {0.0738, 0.3275, 1.465, 4.885, 8.435,
        14.965, 22.485, 43.315};

static const double p1s[] = {-0.897, -0.935, -1.343, -1.372, -1.306,
        -1.305, -1.417, -1.321};

static const double p2s[] = {2.71, 3.23, 6.579, 6.940, 6.253,
        6.155, 7.332, 6.185};

static const double p3s[] = {-0.242, -0.378, -1.186, -1.309, -1.100,
        -1.030, -1.352, -0.983};

OSKAR_EVALUATE_VLA_BEAM_PBCOR_SCALAR(evaluate_vla_beam_pbcor_scalar_float, float, float2)
OSKAR_EVALUATE_VLA_BEAM_PBCOR_SCALAR(evaluate_vla_beam_pbcor_scalar_double, double, double2)
OSKAR_EVALUATE_VLA_BEAM_PBCOR_MATRIX(evaluate_vla_beam_pbcor_matrix_float, float, float4c)
OSKAR_EVALUATE_VLA_BEAM_PBCOR_MATRIX(evaluate_vla_beam_pbcor_matrix_double, double, double4c)

void oskar_evaluate_vla_beam_pbcor(int num_sources, const oskar_Mem* l,
        const oskar_Mem* m, double frequency_hz, oskar_Mem* beam, int* status)
{
    if (*status) return;
    const int precision = oskar_mem_precision(beam);
    const int location = oskar_mem_location(beam);
    if (precision != oskar_mem_type(l) || precision != oskar_mem_type(m))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(l) || location != oskar_mem_location(m))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Find the nearest frequency at which data exists. */
    const double freq_ghz = frequency_hz / 1.0e9;
    const double cutoff_arcmin = 44.376293 / freq_ghz;
    const int index = oskar_find_closest_match_d(freq_ghz,
            sizeof(freqs_ghz) / sizeof(double), freqs_ghz);
    const double p1 = p1s[index];
    const double p2 = p2s[index];
    const double p3 = p3s[index];
    const float freq_ghz_f = (float) freq_ghz;
    const float cutoff_arcmin_f = (float) cutoff_arcmin;
    const float p1_f = (float) p1;
    const float p2_f = (float) p2;
    const float p3_f = (float) p3;
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(beam))
        {
        case OSKAR_SINGLE_COMPLEX:
            evaluate_vla_beam_pbcor_scalar_float(num_sources,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status),
                    freq_ghz_f, p1_f, p2_f, p3_f, cutoff_arcmin_f,
                    oskar_mem_float2(beam, status));
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            evaluate_vla_beam_pbcor_matrix_float(num_sources,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status),
                    freq_ghz_f, p1_f, p2_f, p3_f, cutoff_arcmin_f,
                    oskar_mem_float4c(beam, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            evaluate_vla_beam_pbcor_scalar_double(num_sources,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status),
                    freq_ghz, p1, p2, p3, cutoff_arcmin,
                    oskar_mem_double2(beam, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            evaluate_vla_beam_pbcor_matrix_double(num_sources,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status),
                    freq_ghz, p1, p2, p3, cutoff_arcmin,
                    oskar_mem_double4c(beam, status));
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const int is_dbl = oskar_mem_is_double(beam);
        switch (oskar_mem_type(beam))
        {
        case OSKAR_SINGLE_COMPLEX:
            k = "evaluate_vla_beam_pbcor_scalar_float"; break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "evaluate_vla_beam_pbcor_matrix_float"; break;
        case OSKAR_DOUBLE_COMPLEX:
            k = "evaluate_vla_beam_pbcor_scalar_double"; break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "evaluate_vla_beam_pbcor_matrix_double"; break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_sources, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_sources},
                {PTR_SZ, oskar_mem_buffer_const(l)},
                {PTR_SZ, oskar_mem_buffer_const(m)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&freq_ghz : (const void*)&freq_ghz_f},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (const void*)&p1 : (const void*)&p1_f},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (const void*)&p2 : (const void*)&p2_f},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (const void*)&p3 : (const void*)&p3_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cutoff_arcmin :
                        (const void*)&cutoff_arcmin_f},
                {PTR_SZ, oskar_mem_buffer(beam)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
