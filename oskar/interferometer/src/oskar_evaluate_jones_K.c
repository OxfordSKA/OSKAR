/*
 * Copyright (c) 2011-2019, The University of Oxford
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

#include "interferometer/define_evaluate_jones_K.h"
#include "interferometer/oskar_evaluate_jones_K.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_JONES_K_CPU(evaluate_jones_K_float, float, float2)
OSKAR_JONES_K_CPU(evaluate_jones_K_double, double, double2)

void oskar_evaluate_jones_K(oskar_Jones* K, int num_sources,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        const oskar_Mem* u, const oskar_Mem* v, const oskar_Mem* w,
        double frequency_hz, const oskar_Mem* source_filter,
        double source_filter_min, double source_filter_max, int* status)
{
    if (*status) return;
    const int type = oskar_jones_type(K);
    const int precision = oskar_type_precision(type);
    const int location = oskar_jones_mem_location(K);
    const int num_stations = oskar_jones_num_stations(K);
    const double wavenumber = 2.0 * M_PI * frequency_hz / 299792458.0;
    const float wavenumber_f = (float) wavenumber;
    const float source_filter_min_f = (float) source_filter_min;
    const float source_filter_max_f = (float) source_filter_max;
    if (oskar_mem_location(l) != location ||
            oskar_mem_location(m) != location ||
            oskar_mem_location(n) != location ||
            oskar_mem_location(source_filter) != location ||
            oskar_mem_location(u) != location ||
            oskar_mem_location(v) != location ||
            oskar_mem_location(w) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (precision != oskar_mem_type(l) || precision != oskar_mem_type(m) ||
            precision != oskar_mem_type(n) || precision != oskar_mem_type(u) ||
            precision != oskar_mem_type(v) || precision != oskar_mem_type(w) ||
            precision != oskar_mem_type(source_filter))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE_COMPLEX)
            evaluate_jones_K_float(
                    num_sources,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status),
                    oskar_mem_float_const(n, status),
                    num_stations,
                    oskar_mem_float_const(u, status),
                    oskar_mem_float_const(v, status),
                    oskar_mem_float_const(w, status), wavenumber_f,
                    oskar_mem_float_const(source_filter, status),
                    source_filter_min_f, source_filter_max_f,
                    oskar_jones_float2(K, status));
        else if (type == OSKAR_DOUBLE_COMPLEX)
            evaluate_jones_K_double(
                    num_sources,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status),
                    oskar_mem_double_const(n, status),
                    num_stations,
                    oskar_mem_double_const(u, status),
                    oskar_mem_double_const(v, status),
                    oskar_mem_double_const(w, status), wavenumber,
                    oskar_mem_double_const(source_filter, status),
                    source_filter_min, source_filter_max,
                    oskar_jones_double2(K, status));
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else
    {
        size_t local_size[] = {JONES_K_SOURCE, JONES_K_STATION, 1};
        size_t global_size[] = {1, 1, 1};
        const int is_dbl = (type == OSKAR_DOUBLE_COMPLEX);
        const char* k = 0;
        if (type == OSKAR_SINGLE_COMPLEX)
            k = "evaluate_jones_K_float";
        else if (type == OSKAR_DOUBLE_COMPLEX)
            k = "evaluate_jones_K_double";
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        if (oskar_device_is_cpu(location))
            local_size[1] = 1;
        oskar_device_check_local_size(location, 0, local_size);
        oskar_device_check_local_size(location, 1, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_sources, local_size[0]);
        global_size[1] = oskar_device_global_size(
                (size_t) num_stations, local_size[1]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_sources},
                {PTR_SZ, oskar_mem_buffer_const(l)},
                {PTR_SZ, oskar_mem_buffer_const(m)},
                {PTR_SZ, oskar_mem_buffer_const(n)},
                {INT_SZ, &num_stations},
                {PTR_SZ, oskar_mem_buffer_const(u)},
                {PTR_SZ, oskar_mem_buffer_const(v)},
                {PTR_SZ, oskar_mem_buffer_const(w)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&wavenumber :
                        (const void*)&wavenumber_f},
                {PTR_SZ, oskar_mem_buffer_const(source_filter)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&source_filter_min :
                        (const void*)&source_filter_min_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&source_filter_max :
                        (const void*)&source_filter_max_f},
                {PTR_SZ, oskar_mem_buffer(oskar_jones_mem(K))}
        };
        oskar_device_launch_kernel(k, location, 2, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
