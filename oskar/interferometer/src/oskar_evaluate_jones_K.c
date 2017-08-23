/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include "interferometer/oskar_evaluate_jones_K.h"
#include "interferometer/oskar_evaluate_jones_K_cuda.h"
#include "utility/oskar_device_utils.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_jones_K_f(float2* jones, int num_sources, const float* l,
        const float* m, const float* n, int num_stations,
        const float* u, const float* v, const float* w, float wavenumber,
        const float* source_filter, float source_filter_min,
        float source_filter_max)
{
    int a, s;

    /* Loop over stations. */
    for (a = 0; a < num_stations; ++a)
    {
        float us, vs, ws;
        float2* station_ptr;

        /* Get the station data. */
        station_ptr = &jones[a * num_sources];
        us = wavenumber * u[a];
        vs = wavenumber * v[a];
        ws = wavenumber * w[a];

        /* Loop over sources. */
        for (s = 0; s < num_sources; ++s)
        {
            float phase;
            float2 weight;

            /* Calculate the source phase. */
            if (source_filter[s] > source_filter_min &&
                    source_filter[s] <= source_filter_max)
            {
                phase = us * l[s] + vs * m[s] + ws * (n[s] - 1.0f);
                /* Double precision versions are converted to sincos() by the
                 * compiler, so they're faster and more accurate
                 * than sinf(), cosf(). */
                weight.x = cos(phase);
                weight.y = sin(phase);
            }
            else
            {
                weight.x = 0.0f;
                weight.y = 0.0f;
            }

            /* Store the result. */
            station_ptr[s] = weight;
        }
    }
}

/* Double precision. */
void oskar_evaluate_jones_K_d(double2* jones, int num_sources, const double* l,
        const double* m, const double* n, int num_stations,
        const double* u, const double* v, const double* w, double wavenumber,
        const double* source_filter, double source_filter_min,
        double source_filter_max)
{
    int a, s;

    /* Loop over stations. */
    for (a = 0; a < num_stations; ++a)
    {
        double us, vs, ws;
        double2* station_ptr;

        /* Get the station data. */
        station_ptr = &jones[a * num_sources];
        us = wavenumber * u[a];
        vs = wavenumber * v[a];
        ws = wavenumber * w[a];

        /* Loop over sources. */
        for (s = 0; s < num_sources; ++s)
        {
            double phase;
            double2 weight;

            /* Calculate the source phase. */
            if (source_filter[s] > source_filter_min &&
                    source_filter[s] <= source_filter_max)
            {
                phase = us * l[s] + vs * m[s] + ws * (n[s] - 1.0);
                weight.x = cos(phase);
                weight.y = sin(phase);
            }
            else
            {
                weight.x = 0.0;
                weight.y = 0.0;
            }

            /* Store the result. */
            station_ptr[s] = weight;
        }
    }
}

/* Wrapper. */
void oskar_evaluate_jones_K(oskar_Jones* K, int num_sources,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        const oskar_Mem* u, const oskar_Mem* v, const oskar_Mem* w,
        double frequency_hz, const oskar_Mem* source_filter,
        double source_filter_min, double source_filter_max, int* status)
{
    int num_stations, jones_type, base_type, location;
    double wavenumber;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the Jones matrix block meta-data. */
    jones_type = oskar_jones_type(K);
    base_type = oskar_type_precision(jones_type);
    location = oskar_jones_mem_location(K);
    num_stations = oskar_jones_num_stations(K);
    wavenumber = 2.0 * M_PI * frequency_hz / 299792458.0;

    /* Check that the data is in the right location. */
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

    /* Check that the data are of the right type. */
    if (!oskar_type_is_complex(jones_type) ||
            oskar_type_is_matrix(jones_type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (base_type != oskar_mem_type(l) || base_type != oskar_mem_type(m) ||
            base_type != oskar_mem_type(n) || base_type != oskar_mem_type(u) ||
            base_type != oskar_mem_type(v) || base_type != oskar_mem_type(w) ||
            base_type != oskar_mem_type(source_filter))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Evaluate Jones matrices. */
    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (jones_type == OSKAR_SINGLE_COMPLEX)
        {
            oskar_evaluate_jones_K_cuda_f(oskar_jones_float2(K, status),
                    num_sources,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status),
                    oskar_mem_float_const(n, status),
                    num_stations,
                    oskar_mem_float_const(u, status),
                    oskar_mem_float_const(v, status),
                    oskar_mem_float_const(w, status), wavenumber,
                    oskar_mem_float_const(source_filter, status),
                    source_filter_min, source_filter_max);
        }
        else if (jones_type == OSKAR_DOUBLE_COMPLEX)
        {
            oskar_evaluate_jones_K_cuda_d(oskar_jones_double2(K, status),
                    num_sources,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status),
                    oskar_mem_double_const(n, status),
                    num_stations,
                    oskar_mem_double_const(u, status),
                    oskar_mem_double_const(v, status),
                    oskar_mem_double_const(w, status), wavenumber,
                    oskar_mem_double_const(source_filter, status),
                    source_filter_min, source_filter_max);
        }
        oskar_device_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_CPU)
    {
        if (jones_type == OSKAR_SINGLE_COMPLEX)
        {
            oskar_evaluate_jones_K_f(oskar_jones_float2(K, status),
                    num_sources,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status),
                    oskar_mem_float_const(n, status),
                    num_stations,
                    oskar_mem_float_const(u, status),
                    oskar_mem_float_const(v, status),
                    oskar_mem_float_const(w, status), wavenumber,
                    oskar_mem_float_const(source_filter, status),
                    source_filter_min, source_filter_max);

        }
        else if (jones_type == OSKAR_DOUBLE_COMPLEX)
        {
            oskar_evaluate_jones_K_d(oskar_jones_double2(K, status),
                    num_sources,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status),
                    oskar_mem_double_const(n, status),
                    num_stations,
                    oskar_mem_double_const(u, status),
                    oskar_mem_double_const(v, status),
                    oskar_mem_double_const(w, status), wavenumber,
                    oskar_mem_double_const(source_filter, status),
                    source_filter_min, source_filter_max);
        }
    }
}

#ifdef __cplusplus
}
#endif
