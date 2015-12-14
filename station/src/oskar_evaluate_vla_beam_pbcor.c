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

#include <math.h>

#include <oskar_evaluate_vla_beam_pbcor.h>
#include <oskar_evaluate_vla_beam_pbcor_cuda.h>
#include <oskar_cuda_check_error.h>
#include <oskar_find_closest_match.h>
#include <oskar_vla_pbcor_inline.h>

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


void oskar_evaluate_vla_beam_pbcor_f(float* beam, int num_sources,
        const float* l, const float* m, const float freq_ghz, const float p1,
        const float p2, const float p3)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        beam[i] = oskar_vla_pbcor_inline_f(l[i], m[i], freq_ghz, p1, p2, p3);
    }
}


void oskar_evaluate_vla_beam_pbcor_complex_f(float2* beam, int num_sources,
        const float* l, const float* m, const float freq_ghz, const float p1,
        const float p2, const float p3)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        beam[i].x = oskar_vla_pbcor_inline_f(l[i], m[i], freq_ghz, p1, p2, p3);
        beam[i].y = 0.0f;
    }
}


void oskar_evaluate_vla_beam_pbcor_matrix_f(float4c* beam, int num_sources,
        const float* l, const float* m, const float freq_ghz, const float p1,
        const float p2, const float p3)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        float t;
        t = oskar_vla_pbcor_inline_f(l[i], m[i], freq_ghz, p1, p2, p3);
        beam[i].a.x = t;
        beam[i].a.y = 0.0f;
        beam[i].b.x = 0.0f;
        beam[i].b.y = 0.0f;
        beam[i].c.x = 0.0f;
        beam[i].c.y = 0.0f;
        beam[i].d.x = t;
        beam[i].d.y = 0.0f;
    }
}


void oskar_evaluate_vla_beam_pbcor_d(double* beam, int num_sources,
        const double* l, const double* m, const double freq_ghz,
        const double p1, const double p2, const double p3)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        beam[i] = oskar_vla_pbcor_inline_d(l[i], m[i], freq_ghz, p1, p2, p3);
    }
}


void oskar_evaluate_vla_beam_pbcor_complex_d(double2* beam, int num_sources,
        const double* l, const double* m, const double freq_ghz,
        const double p1, const double p2, const double p3)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        beam[i].x = oskar_vla_pbcor_inline_d(l[i], m[i], freq_ghz, p1, p2, p3);
        beam[i].y = 0.0;
    }
}


void oskar_evaluate_vla_beam_pbcor_matrix_d(double4c* beam, int num_sources,
        const double* l, const double* m, const double freq_ghz,
        const double p1, const double p2, const double p3)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        double t;
        t = oskar_vla_pbcor_inline_d(l[i], m[i], freq_ghz, p1, p2, p3);
        beam[i].a.x = t;
        beam[i].a.y = 0.0;
        beam[i].b.x = 0.0;
        beam[i].b.y = 0.0;
        beam[i].c.x = 0.0;
        beam[i].c.y = 0.0;
        beam[i].d.x = t;
        beam[i].d.y = 0.0;
    }
}


void oskar_evaluate_vla_beam_pbcor(oskar_Mem* beam, int num_sources,
        const oskar_Mem* l, const oskar_Mem* m, double frequency_hz,
        int* status)
{
    int index, precision, type, location;
    double f, p1, p2, p3;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type and location. */
    precision = oskar_mem_precision(beam);
    type = oskar_mem_type(beam);
    location = oskar_mem_location(beam);
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
    index = oskar_find_closest_match_d(frequency_hz / 1.0e9,
            sizeof(freqs_ghz) / sizeof(double), freqs_ghz);
    f = frequency_hz / 1.0e9;
    p1 = p1s[index];
    p2 = p2s[index];
    p3 = p3s[index];

    /* Switch on precision. */
    if (precision == OSKAR_SINGLE)
    {
        const float *l_, *m_;
        l_ = oskar_mem_float_const(l, status);
        m_ = oskar_mem_float_const(m, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            if (type == OSKAR_SINGLE)
            {
                oskar_evaluate_vla_beam_pbcor_cuda_f(
                        oskar_mem_float(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
            else if (type == OSKAR_SINGLE_COMPLEX)
            {
                oskar_evaluate_vla_beam_pbcor_complex_cuda_f(
                        oskar_mem_float2(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
            else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
            {
                oskar_evaluate_vla_beam_pbcor_matrix_cuda_f(
                        oskar_mem_float4c(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            if (type == OSKAR_SINGLE)
            {
                oskar_evaluate_vla_beam_pbcor_f(
                        oskar_mem_float(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
            else if (type == OSKAR_SINGLE_COMPLEX)
            {
                oskar_evaluate_vla_beam_pbcor_complex_f(
                        oskar_mem_float2(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
            else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
            {
                oskar_evaluate_vla_beam_pbcor_matrix_f(
                        oskar_mem_float4c(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
        }
    }
    else if (precision == OSKAR_DOUBLE)
    {
        const double *l_, *m_;
        l_ = oskar_mem_double_const(l, status);
        m_ = oskar_mem_double_const(m, status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            if (type == OSKAR_DOUBLE)
            {
                oskar_evaluate_vla_beam_pbcor_cuda_d(
                        oskar_mem_double(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
            else if (type == OSKAR_DOUBLE_COMPLEX)
            {
                oskar_evaluate_vla_beam_pbcor_complex_cuda_d(
                        oskar_mem_double2(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
            else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
            {
                oskar_evaluate_vla_beam_pbcor_matrix_cuda_d(
                        oskar_mem_double4c(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_CPU)
        {
            if (type == OSKAR_DOUBLE)
            {
                oskar_evaluate_vla_beam_pbcor_d(
                        oskar_mem_double(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
            else if (type == OSKAR_DOUBLE_COMPLEX)
            {
                oskar_evaluate_vla_beam_pbcor_complex_d(
                        oskar_mem_double2(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
            else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
            {
                oskar_evaluate_vla_beam_pbcor_matrix_d(
                        oskar_mem_double4c(beam, status),
                        num_sources, l_, m_, f, p1, p2, p3);
            }
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
