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

#include <oskar_sky.h>
#include <oskar_scale_flux_with_frequency_cuda.h>
#include <oskar_scale_flux_with_frequency.h>
#include <oskar_cuda_check_error.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_scale_flux_with_frequency(oskar_Sky* model, double frequency,
        int* status)
{
    int type, location, num_sources;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the type, location and dimensions. */
    type = oskar_sky_precision(model);
    location = oskar_sky_mem_location(model);
    num_sources = oskar_sky_num_sources(model);

    /* Scale the flux values. */
    if (type == OSKAR_SINGLE)
    {
        float *I, *Q, *U, *V, *ref;
        const float *spix, *rm;
        I    = oskar_mem_float(oskar_sky_I(model), status);
        Q    = oskar_mem_float(oskar_sky_Q(model), status);
        U    = oskar_mem_float(oskar_sky_U(model), status);
        V    = oskar_mem_float(oskar_sky_V(model), status);
        ref  = oskar_mem_float(oskar_sky_reference_freq_hz(model), status);
        spix = oskar_mem_float_const(
                oskar_sky_spectral_index_const(model), status);
        rm   = oskar_mem_float_const(
                oskar_sky_rotation_measure_rad_const(model), status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_scale_flux_with_frequency_cuda_f(num_sources, frequency,
                    I, Q, U, V, ref, spix, rm);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_scale_flux_with_frequency_f(num_sources, frequency,
                    I, Q, U, V, ref, spix, rm);
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        double *I, *Q, *U, *V, *ref;
        const double *spix, *rm;
        I    = oskar_mem_double(oskar_sky_I(model), status);
        Q    = oskar_mem_double(oskar_sky_Q(model), status);
        U    = oskar_mem_double(oskar_sky_U(model), status);
        V    = oskar_mem_double(oskar_sky_V(model), status);
        ref  = oskar_mem_double(oskar_sky_reference_freq_hz(model), status);
        spix = oskar_mem_double_const(
                oskar_sky_spectral_index_const(model), status);
        rm   = oskar_mem_double_const(
                oskar_sky_rotation_measure_rad_const(model), status);

        if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_scale_flux_with_frequency_cuda_d(num_sources, frequency,
                    I, Q, U, V, ref, spix, rm);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
        {
            oskar_scale_flux_with_frequency_d(num_sources, frequency,
                    I, Q, U, V, ref, spix, rm);
        }
    }
}

#ifdef __cplusplus
}
#endif
