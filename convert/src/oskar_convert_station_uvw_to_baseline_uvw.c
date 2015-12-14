/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include <oskar_convert_station_uvw_to_baseline_uvw.h>
#include <oskar_convert_station_uvw_to_baseline_uvw_cuda.h>
#include <oskar_cuda_check_error.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_station_uvw_to_baseline_uvw_f(int num_stations,
        const float* u, const float* v, const float* w, float* uu,
        float* vv, float* ww)
{
    int s1, s2, b; /* Station and baseline indices. */
    for (s1 = 0, b = 0; s1 < num_stations; ++s1)
    {
        for (s2 = s1 + 1; s2 < num_stations; ++s2, ++b)
        {
            uu[b] = u[s2] - u[s1];
            vv[b] = v[s2] - v[s1];
            ww[b] = w[s2] - w[s1];
        }
    }
}

/* Double precision. */
void oskar_convert_station_uvw_to_baseline_uvw_d(int num_stations,
        const double* u, const double* v, const double* w, double* uu,
        double* vv, double* ww)
{
    int s1, s2, b; /* Station and baseline indices. */
    for (s1 = 0, b = 0; s1 < num_stations; ++s1)
    {
        for (s2 = s1 + 1; s2 < num_stations; ++s2, ++b)
        {
            uu[b] = u[s2] - u[s1];
            vv[b] = v[s2] - v[s1];
            ww[b] = w[s2] - w[s1];
        }
    }
}

/* Wrapper. */
void oskar_convert_station_uvw_to_baseline_uvw(const oskar_Mem* u,
        const oskar_Mem* v, const oskar_Mem* w, oskar_Mem* uu, oskar_Mem* vv,
        oskar_Mem* ww, int* status)
{
    int type, location, num_stations, num_baselines;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get data type, location and size. */
    type = oskar_mem_type(u);
    location = oskar_mem_location(u);
    num_stations = (int)oskar_mem_length(u);
    num_baselines = num_stations * (num_stations - 1) / 2;

    /* Check that the data types match. */
    if (oskar_mem_type(v) != type || oskar_mem_type(w) != type ||
            oskar_mem_type(uu) != type || oskar_mem_type(vv) != type ||
            oskar_mem_type(ww) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check that the data locations match. */
    if (oskar_mem_location(v) != location ||
            oskar_mem_location(w) != location ||
            oskar_mem_location(uu) != location ||
            oskar_mem_location(vv) != location ||
            oskar_mem_location(ww) != location)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check that the memory is allocated. */
    if (!oskar_mem_allocated(uu) || !oskar_mem_allocated(vv) ||
            !oskar_mem_allocated(ww) || !oskar_mem_allocated(u) ||
            !oskar_mem_allocated(v) || !oskar_mem_allocated(w))
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Check that the data dimensions are OK. */
    if ((int)oskar_mem_length(v) < num_stations ||
            (int)oskar_mem_length(w) < num_stations ||
            (int)oskar_mem_length(uu) < num_baselines ||
            (int)oskar_mem_length(vv) < num_baselines ||
            (int)oskar_mem_length(ww) < num_baselines)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_convert_station_uvw_to_baseline_uvw_f(num_stations,
                    oskar_mem_float_const(u, status),
                    oskar_mem_float_const(v, status),
                    oskar_mem_float_const(w, status),
                    oskar_mem_float(uu, status),
                    oskar_mem_float(vv, status),
                    oskar_mem_float(ww, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_convert_station_uvw_to_baseline_uvw_d(num_stations,
                    oskar_mem_double_const(u, status),
                    oskar_mem_double_const(v, status),
                    oskar_mem_double_const(w, status),
                    oskar_mem_double(uu, status),
                    oskar_mem_double(vv, status),
                    oskar_mem_double(ww, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
        {
            oskar_convert_station_uvw_to_baseline_uvw_cuda_f(num_stations,
                    oskar_mem_float_const(u, status),
                    oskar_mem_float_const(v, status),
                    oskar_mem_float_const(w, status),
                    oskar_mem_float(uu, status),
                    oskar_mem_float(vv, status),
                    oskar_mem_float(ww, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_convert_station_uvw_to_baseline_uvw_cuda_d(num_stations,
                    oskar_mem_double_const(u, status),
                    oskar_mem_double_const(v, status),
                    oskar_mem_double_const(w, status),
                    oskar_mem_double(uu, status),
                    oskar_mem_double(vv, status),
                    oskar_mem_double(ww, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
        oskar_cuda_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
}

#ifdef __cplusplus
}
#endif
