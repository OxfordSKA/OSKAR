/*
 * Copyright (c) 2012, The University of Oxford
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

#include "sky/oskar_sky_model_location.h"
#include "sky/oskar_sky_model_type.h"
#include "interferometry/oskar_correlate.h"
#include "interferometry/cudak/oskar_cudak_correlator_scalar.h"
#include "interferometry/cudak/oskar_cudak_correlator.h"
#include "interferometry/cudak/oskar_cudak_correlator_extended.h"
#include "interferometry/cudak/oskar_cudak_correlator_time_smearing_extended.h"
#include "interferometry/cudak/oskar_cudak_correlator_time_smearing.h"
#include "utility/oskar_cuda_check_error.h"
#include "utility/oskar_mem_type_check.h"
#include <stdio.h>

#define C_0 299792458.0

extern "C"
void oskar_correlate(oskar_Mem* vis, const oskar_Jones* J,
        const oskar_TelescopeModel* telescope, const oskar_SkyModel* sky,
        const oskar_Mem* u, const oskar_Mem* v, double gast, int* status)
{
    int base_type, location, n_stations, n_sources;

    /* Check all inputs. */
    if (!vis || !J || !telescope || !sky || !u || !v || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data dimensions. */
    n_stations = telescope->num_stations;
    n_sources = sky->num_sources;

    /* Check data locations. */
    location = oskar_sky_model_location(sky);
    if (location != OSKAR_LOCATION_GPU)
        *status = OSKAR_ERR_BAD_LOCATION;
    if (vis->location != location || J->data.location != location ||
            u->location != location || v->location != location ||
            telescope->station_x.location != location ||
            telescope->station_y.location != location)
        *status = OSKAR_ERR_LOCATION_MISMATCH;

    /* Check for consistent data types. */
    base_type = oskar_sky_model_type(sky);
    if (oskar_mem_base_type(vis->type) != base_type ||
            oskar_mem_base_type(J->data.type) != base_type ||
            u->type != base_type || v->type != base_type)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* If neither single or double precision, return error. */
    if (base_type != OSKAR_SINGLE && base_type != OSKAR_DOUBLE)
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    /* Check the input dimensions. */
    if (J->num_sources != n_sources || u->num_elements != n_stations ||
            v->num_elements != n_stations)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check there is enough space for the result. */
    if (vis->num_elements < n_stations * (n_stations - 1) / 2)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get bandwidth-smearing term. */
    double lambda_bandwidth = telescope->wavelength_metres *
            telescope->bandwidth_hz;
    double freq = C_0 / telescope->wavelength_metres;
    double bandwidth = telescope->bandwidth_hz;

    /* Get time-average smearing term and Greenwich hour angle. */
    double time_avg = telescope->time_average_sec;
    double gha0 = gast - telescope->ra0_rad;

    /* Check type of Jones matrix. */
    if (oskar_mem_is_matrix(J->data.type) && oskar_mem_is_matrix(vis->type))
    {
        /* Call the kernel for full polarisation. */
        if (base_type == OSKAR_DOUBLE)
        {
            dim3 num_threads(128, 1);
            dim3 num_blocks(n_stations, n_stations);
            size_t shared_mem = num_threads.x * sizeof(double4c);

            if (time_avg > 0.0)
            {
                if (sky->use_extended)
                {
                    oskar_cudak_correlator_time_smearing_extended_d
                    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                    (n_sources, n_stations, J->data, sky->I, sky->Q, sky->U,
                            sky->V, sky->rel_l, sky->rel_m, sky->rel_n,
                            sky->gaussian_a, sky->gaussian_b, sky->gaussian_c,
                            *u, *v, telescope->station_x, telescope->station_y,
                            freq, bandwidth, time_avg, gha0,
                            telescope->dec0_rad, *vis);
                }
                else
                {
                    oskar_cudak_correlator_time_smearing_d
                    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                    (n_sources, n_stations, J->data, sky->I, sky->Q, sky->U,
                            sky->V, sky->rel_l, sky->rel_m, sky->rel_n,
                            *u, *v, telescope->station_x, telescope->station_y,
                            freq, bandwidth, time_avg, gha0,
                            telescope->dec0_rad, *vis);
                }
            }
            else
            {
                if (sky->use_extended)
                {
                    oskar_cudak_correlator_extended_d
                    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                    (n_sources, n_stations, J->data, sky->I, sky->Q, sky->U,
                            sky->V, *u, *v, sky->rel_l, sky->rel_m,
                            freq, bandwidth, sky->gaussian_a, sky->gaussian_b,
                            sky->gaussian_c, *vis);
                }
                else
                {
                    oskar_cudak_correlator_d
                    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                    (n_sources, n_stations, J->data, sky->I, sky->Q, sky->U, sky->V,
                            *u, *v, sky->rel_l, sky->rel_m, lambda_bandwidth, *vis);
                }
            }
        }
        else if (base_type == OSKAR_SINGLE)
        {
            dim3 num_threads(128, 1);
            dim3 num_blocks(n_stations, n_stations);
            size_t shared_mem = num_threads.x * sizeof(float4c);

            if (time_avg > 0.0)
            {
                if (sky->use_extended)
                {
                    oskar_cudak_correlator_time_smearing_extended_f
                    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                    (n_sources, n_stations, J->data, sky->I, sky->Q, sky->U,
                            sky->V, sky->rel_l, sky->rel_m, sky->rel_n,
                            sky->gaussian_a, sky->gaussian_b, sky->gaussian_c,
                            *u, *v, telescope->station_x, telescope->station_y,
                            freq, bandwidth, time_avg, gha0,
                            telescope->dec0_rad, *vis);
                }
                else
                {
                    oskar_cudak_correlator_time_smearing_f
                    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                    (n_sources, n_stations, J->data, sky->I, sky->Q, sky->U,
                            sky->V, sky->rel_l, sky->rel_m, sky->rel_n,
                            *u, *v, telescope->station_x, telescope->station_y,
                            freq, bandwidth, time_avg, gha0,
                            telescope->dec0_rad, *vis);
                }
            }
            else
            {
                if (sky->use_extended)
                {
                    oskar_cudak_correlator_extended_f
                    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                    (n_sources, n_stations, J->data, sky->I, sky->Q, sky->U,
                            sky->V, *u, *v, sky->rel_l, sky->rel_m,
                            freq, bandwidth, sky->gaussian_a, sky->gaussian_b,
                            sky->gaussian_c, *vis);
                }
                else
                {
                    oskar_cudak_correlator_f
                    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                    (n_sources, n_stations, J->data, sky->I, sky->Q, sky->U, sky->V,
                            *u, *v, sky->rel_l, sky->rel_m, lambda_bandwidth, *vis);
                }
            }
        }
    }

    /* Jones type --> Scalar version. */
    else
    {
        if (sky->use_extended)
        {
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            return;
        }

        /* Call the kernel for scalar simulation. */
        if (base_type == OSKAR_DOUBLE)
        {
            dim3 num_threads(128, 1);
            dim3 num_blocks(n_stations, n_stations);
            size_t shared_mem = num_threads.x * sizeof(double2);

            oskar_cudak_correlator_scalar_d
                OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                (n_sources, n_stations, J->data, sky->I, *u, *v, sky->rel_l,
                        sky->rel_m, lambda_bandwidth, *vis);
        }
        else if (base_type == OSKAR_SINGLE)
        {
            dim3 num_threads(128, 1);
            dim3 num_blocks(n_stations, n_stations);
            size_t shared_mem = num_threads.x * sizeof(float2);

            oskar_cudak_correlator_scalar_f
                OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
                (n_sources, n_stations, J->data, sky->I, *u, *v, sky->rel_l,
                        sky->rel_m, lambda_bandwidth, *vis);
        }
    }

    oskar_cuda_check_error(status);
}
