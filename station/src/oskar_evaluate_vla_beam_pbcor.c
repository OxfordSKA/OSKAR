/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_evaluate_vla_beam_pbcor.h>
#include <oskar_evaluate_vla_beam_pbcor_cuda.h>
#include <oskar_cuda_check_error.h>
#include <oskar_find_closest_match.h>

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

static const double cutoff_freqs_ghz[] = {1.485, 4.885, 15.0, 22.5};
static const double cutoff_radii_arcmin[] = {29.8, 9.13, 2.95, 1.97};


void oskar_evaluate_vla_beam_pbcor_f(float* beam, int num_sources,
        const float* radius_arcmin, const float freq_ghz, const float p1,
        const float p2, const float p3, const float cutoff_radius_arcmin)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        float r, t, X;
        r = radius_arcmin[i];
        if (r < cutoff_radius_arcmin)
        {
            t = r * freq_ghz;
            X = t * t;
            beam[i] = 1.0f +
                    X * (p1 * 1e-3f + X * (p2 * 1e-7f + X * p3 * 1e-10f));
        }
        else
        {
            beam[i] = 0.0f;
        }
    }
}


void oskar_evaluate_vla_beam_pbcor_d(double* beam, int num_sources,
        const double* radius_arcmin, const double freq_ghz, const double p1,
        const double p2, const double p3, const double cutoff_radius_arcmin)
{
    int i;
    for (i = 0; i < num_sources; ++i)
    {
        double r, t, X;
        r = radius_arcmin[i];
        if (r < cutoff_radius_arcmin)
        {
            t = r * freq_ghz;
            X = t * t;
            beam[i] = 1.0 + X * (p1 * 1e-3 + X * (p2 * 1e-7 + X * p3 * 1e-10));
        }
        else
        {
            beam[i] = 0.0;
        }
    }
}


void oskar_evaluate_vla_beam_pbcor(oskar_Mem* beam, int num_sources,
        const oskar_Mem* radius_arcmin, double frequency_hz, int* status)
{
    int index, type, location;
    double f, p1, p2, p3, cutoff_radius;

    /* Check all inputs. */
    if (!status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type and location. */
    type = oskar_mem_type(beam);
    location = oskar_mem_location(beam);
    if (type != oskar_mem_type(radius_arcmin))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(radius_arcmin))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Find the nearest frequency at which data exists. */
    index = oskar_find_closest_match_d(frequency_hz / 1.0e9,
            sizeof(freqs_ghz) / sizeof(double), freqs_ghz);
    f = freqs_ghz[index];
    p1 = p1s[index];
    p2 = p2s[index];
    p3 = p3s[index];

    /* Find the nearest cutoff radius for the given frequency. */
    index = oskar_find_closest_match_d(frequency_hz / 1.0e9,
            sizeof(cutoff_freqs_ghz) / sizeof(double), cutoff_freqs_ghz);
    cutoff_radius = cutoff_radii_arcmin[index];

    /* Switch on type. */
    if (type == OSKAR_SINGLE)
    {
        float* beam_;
        const float* radius_;
        beam_ = oskar_mem_float(beam, status);
        radius_ = oskar_mem_float_const(beam, status);

        if (location == OSKAR_LOCATION_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_vla_beam_pbcor_cuda_f(beam_, num_sources, radius_,
                    f, p1, p2, p3, cutoff_radius);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_LOCATION_CPU)
        {
            oskar_evaluate_vla_beam_pbcor_f(beam_, num_sources, radius_,
                    f, p1, p2, p3, cutoff_radius);
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        double* beam_;
        const double* radius_;
        beam_ = oskar_mem_double(beam, status);
        radius_ = oskar_mem_double_const(beam, status);

        if (location == OSKAR_LOCATION_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_evaluate_vla_beam_pbcor_cuda_d(beam_, num_sources, radius_,
                    f, p1, p2, p3, cutoff_radius);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else if (location == OSKAR_LOCATION_CPU)
        {
            oskar_evaluate_vla_beam_pbcor_d(beam_, num_sources, radius_,
                    f, p1, p2, p3, cutoff_radius);
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
