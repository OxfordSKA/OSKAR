/*
 * Copyright (c) 2011, The University of Oxford
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

#include "interferometry/oskar_correlate.h"
#include "interferometry/cudak/oskar_cudak_correlator_scalar.h"
#include "interferometry/cudak/oskar_cudak_correlator.h"
#include "interferometry/cudak/oskar_cudak_correlator_extended.h"
#include <stdio.h>

#define C_0 299792458.0

extern "C"
int oskar_correlate(oskar_Mem* vis, const oskar_Jones* J,
        const oskar_TelescopeModel* telescope, const oskar_SkyModel* sky,
        const oskar_Mem* u, const oskar_Mem* v)
{
    // Type flags.
    bool single_precision = false, double_precision = false;

    // Check data location.
    if (vis->location() != OSKAR_LOCATION_GPU ||
            J->location() != OSKAR_LOCATION_GPU ||
            sky->location() != OSKAR_LOCATION_GPU ||
            u->location() != OSKAR_LOCATION_GPU ||
            v->location() != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Check if single precision.
    single_precision = (vis->is_single() && J->ptr.is_single() &&
            sky->I.is_single() && sky->Q.is_single() && sky->U.is_single() &&
            sky->V.is_single() && sky->rel_l.is_single() &&
            sky->rel_m.is_single() && u->is_single() && v->is_single());

    // If not single precision, check if double precision.
    if (!single_precision)
        double_precision = (vis->is_double() && J->ptr.is_double() &&
                sky->I.is_double() && sky->Q.is_double() &&
                sky->U.is_double() && sky->V.is_double() &&
                sky->rel_l.is_double() && sky->rel_m.is_double() &&
                u->is_double() && v->is_double());

    // If neither single or double precision, return error.
    if (!single_precision && !double_precision)
        return OSKAR_ERR_BAD_DATA_TYPE;

    // Check the input dimensions.
    int n_stations = telescope->num_stations;
    int n_sources = sky->num_sources;
    if (J->num_sources() != n_sources || u->num_elements() != n_stations ||
            v->num_elements() != n_stations)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    // Check there is enough space for the result.
    if (vis->num_elements() < n_stations * (n_stations - 1) / 2)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    // Get bandwidth-smearing term.
    double lambda_bandwidth = telescope->wavelength_metres *
            telescope->bandwidth_hz;
    double freq = C_0 / telescope->wavelength_metres;
    double bandwidth = telescope->bandwidth_hz;

    // Check type of Jones matrix.
    if (J->ptr.is_matrix() && vis->is_matrix())
    {
        // Call the kernel for full polarisation.
        if (double_precision)
        {
            dim3 num_threads(128, 1); // Antennas, antennas.
            dim3 num_blocks(n_stations, n_stations);
            size_t shared_mem = num_threads.x * sizeof(double4c);

            if (sky->use_extended)
            {
                oskar_cudak_correlator_extended_d
                    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                    (n_sources, n_stations, J->ptr, sky->I, sky->Q, sky->U,
                            sky->V, *u, *v, sky->rel_l, sky->rel_m,
                            freq, bandwidth, sky->gaussian_a, sky->gaussian_b,
                            sky->gaussian_c, *vis);
            }
            else
            {
                oskar_cudak_correlator_d
                OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                (n_sources, n_stations, J->ptr, sky->I, sky->Q, sky->U, sky->V,
                        *u, *v, sky->rel_l, sky->rel_m, lambda_bandwidth, *vis);
            }
        }
        else
        {
            dim3 num_threads(128, 1); // Antennas, antennas.
            dim3 num_blocks(n_stations, n_stations);
            size_t shared_mem = num_threads.x * sizeof(float4c);

            if (sky->use_extended)
            {
                oskar_cudak_correlator_extended_f
                    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                    (n_sources, n_stations, J->ptr, sky->I, sky->Q, sky->U,
                            sky->V, *u, *v, sky->rel_l, sky->rel_m,
                            freq, bandwidth, sky->gaussian_a, sky->gaussian_b,
                            sky->gaussian_c, *vis);
            }
            else
            {
                oskar_cudak_correlator_f
                    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                    (n_sources, n_stations, J->ptr, sky->I, sky->Q, sky->U, sky->V,
                            *u, *v, sky->rel_l, sky->rel_m, lambda_bandwidth, *vis);
            }
        }
    }
    else
    {
        if (sky->use_extended)
            return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;

        // Call the kernel for scalar simulation.
        if (double_precision)
        {
            dim3 num_threads(128, 1); // Antennas, antennas.
            dim3 num_blocks(n_stations, n_stations);
            size_t shared_mem = num_threads.x * sizeof(double2);

            oskar_cudak_correlator_scalar_d
                OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                (n_sources, n_stations, J->ptr, sky->I, *u, *v, sky->rel_l,
                        sky->rel_m, lambda_bandwidth, *vis);
        }
        else
        {
            dim3 num_threads(128, 1); // Antennas, antennas.
            dim3 num_blocks(n_stations, n_stations);
            size_t shared_mem = num_threads.x * sizeof(float2);

            oskar_cudak_correlator_scalar_f
                OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                (n_sources, n_stations, J->ptr, sky->I, *u, *v, sky->rel_l,
                        sky->rel_m, lambda_bandwidth, *vis);
        }
    }

    cudaDeviceSynchronize();
    return cudaPeekAtLastError();
}
