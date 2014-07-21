/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <oskar_evaluate_jones_K.h>
#include <oskar_evaluate_jones_K_cuda.h>
#include <oskar_cuda_check_error.h>
#include <oskar_cmath.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_jones_K_f(float2* jones, int num_sources, const float* l,
        const float* m, const float* n, int num_stations,
        const float* u, const float* v, const float* w, float wavenumber)
{
    int station, source;

    /* Loop over stations. */
    for (station = 0; station < num_stations; ++station)
    {
        float us, vs, ws;
        float2* station_ptr;

        /* Get the station data. */
        station_ptr = &jones[station * num_sources];
        us = wavenumber * u[station];
        vs = wavenumber * v[station];
        ws = wavenumber * w[station];

        /* Loop over sources. */
        for (source = 0; source < num_sources; ++source)
        {
            float phase;
            float2 weight;

            /* Calculate the source phase. */
            phase = us * l[source] + vs * m[source] + ws * (n[source] - 1.0f);
            weight.x = cosf(phase);
            weight.y = sinf(phase);

            /* Store the result. */
            station_ptr[source] = weight;
        }
    }
}

/* Double precision. */
void oskar_evaluate_jones_K_d(double2* jones, int num_sources, const double* l,
        const double* m, const double* n, int num_stations,
        const double* u, const double* v, const double* w, double wavenumber)
{
    int station, source;

    /* Loop over stations. */
    for (station = 0; station < num_stations; ++station)
    {
        double us, vs, ws;
        double2* station_ptr;

        /* Get the station data. */
        station_ptr = &jones[station * num_sources];
        us = wavenumber * u[station];
        vs = wavenumber * v[station];
        ws = wavenumber * w[station];

        /* Loop over sources. */
        for (source = 0; source < num_sources; ++source)
        {
            double phase;
            double2 weight;

            /* Calculate the source phase. */
            phase = us * l[source] + vs * m[source] + ws * (n[source] - 1.0);
            weight.x = cos(phase);
            weight.y = sin(phase);

            /* Store the result. */
            station_ptr[source] = weight;
        }
    }
}

/* Wrapper. */
void oskar_evaluate_jones_K(oskar_Jones* K, int num_sources,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        const oskar_Mem* u, const oskar_Mem* v, const oskar_Mem* w,
        double frequency_hz, int* status)
{
    int num_stations, jones_type, base_type, location;
    double wavenumber;

    /* Check all inputs. */
    if (!K || !l || !m || !n || !u || !v || !w || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the Jones matrix block meta-data. */
    jones_type = oskar_jones_type(K);
    base_type = oskar_mem_type_precision(jones_type);
    location = oskar_jones_mem_location(K);
    num_stations = oskar_jones_num_stations(K);
    wavenumber = 2.0 * M_PI * frequency_hz / 299792458.0;

    /* Check that the data dimensions are OK. */
    if (num_sources > (int)oskar_mem_length(l) ||
            num_sources > (int)oskar_mem_length(m) ||
            num_sources > (int)oskar_mem_length(n) ||
            num_stations != (int)oskar_mem_length(u) ||
            num_stations != (int)oskar_mem_length(v) ||
            num_stations != (int)oskar_mem_length(w))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check that the data is in the right location. */
    if (oskar_mem_location(l) != location ||
            oskar_mem_location(m) != location ||
            oskar_mem_location(n) != location ||
            oskar_mem_location(u) != location ||
            oskar_mem_location(v) != location ||
            oskar_mem_location(w) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check that the data are of the right type. */
    if (!oskar_mem_type_is_complex(jones_type) ||
            oskar_mem_type_is_matrix(jones_type))
    {
        *status = OSKAR_ERR_BAD_JONES_TYPE;
        return;
    }
    if (base_type != oskar_mem_type(l) || base_type != oskar_mem_type(m) ||
            base_type != oskar_mem_type(n) || base_type != oskar_mem_type(u) ||
            base_type != oskar_mem_type(v) || base_type != oskar_mem_type(w))
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
                    oskar_mem_float_const(w, status), wavenumber);
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
                    oskar_mem_double_const(w, status), wavenumber);
        }
        oskar_cuda_check_error(status);
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
                    oskar_mem_float_const(w, status), wavenumber);
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
                    oskar_mem_double_const(w, status), wavenumber);
        }
    }
}

#ifdef __cplusplus
}
#endif
