/*
 * Copyright (c) 2011-2013, The University of Oxford
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
#include "interferometry/oskar_correlate_extended_cuda.h"
#include "interferometry/oskar_correlate_extended_time_smearing_cuda.h"
#include "interferometry/oskar_correlate_point_cuda.h"
#include "interferometry/oskar_correlate_point_scalar_cuda.h"
#include "interferometry/oskar_correlate_point_time_smearing_cuda.h"
#include "utility/oskar_cuda_check_error.h"
#include "utility/oskar_mem_type_check.h"
#include <stdio.h>

#define C_0 299792458.0

#ifdef __cplusplus
extern "C" {
#endif

void oskar_correlate(oskar_Mem* vis, const oskar_Jones* J,
        const oskar_TelescopeModel* telescope, const oskar_SkyModel* sky,
        const oskar_Mem* u, const oskar_Mem* v, double gast, int* status)
{
    int base_type, location, n_stations, n_sources;
    double bandwidth, lambda_bandwidth, freq, time_avg, gha0;

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

    /* Get bandwidth-smearing term. */
    bandwidth = telescope->bandwidth_hz;
    lambda_bandwidth = telescope->wavelength_metres * bandwidth;
    freq = C_0 / telescope->wavelength_metres;

    /* Get time-average smearing term and Greenwich hour angle. */
    time_avg = telescope->time_average_sec;
    gha0 = gast - telescope->ra0_rad;

    /* Check data locations. */
    location = oskar_sky_model_location(sky);
    if (vis->location != location || J->data.location != location ||
            u->location != location || v->location != location ||
            telescope->station_x.location != location ||
            telescope->station_y.location != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check for consistent data types. */
    base_type = oskar_sky_model_type(sky);
    if (oskar_mem_base_type(vis->type) != base_type ||
            oskar_mem_base_type(J->data.type) != base_type ||
            u->type != base_type || v->type != base_type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (vis->type != J->data.type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* If neither single or double precision, return error. */
    if (base_type != OSKAR_SINGLE && base_type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Check the input dimensions. */
    if (J->num_sources != n_sources || u->num_elements != n_stations ||
            v->num_elements != n_stations)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check there is enough space for the result. */
    if (vis->num_elements < n_stations * (n_stations - 1) / 2)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check if memory is on the device. */
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        /* Check if Jones type is a matrix. */
        if (oskar_mem_is_matrix(J->data.type) && oskar_mem_is_matrix(vis->type))
        {
            if (base_type == OSKAR_DOUBLE)
            {
                if (time_avg > 0.0)
                {
                    if (sky->use_extended)
                    {
                        oskar_correlate_extended_time_smearing_cuda_d
                        (n_sources, n_stations, (const double4c*)J->data.data,
                                (const double*)sky->I.data,
                                (const double*)sky->Q.data,
                                (const double*)sky->U.data,
                                (const double*)sky->V.data,
                                (const double*)sky->rel_l.data,
                                (const double*)sky->rel_m.data,
                                (const double*)sky->rel_n.data,
                                (const double*)sky->gaussian_a.data,
                                (const double*)sky->gaussian_b.data,
                                (const double*)sky->gaussian_c.data,
                                (const double*)u->data, (const double*)v->data,
                                (const double*)telescope->station_x.data,
                                (const double*)telescope->station_y.data,
                                freq, bandwidth, time_avg, gha0,
                                telescope->dec0_rad, (double4c*)vis->data);
                    }
                    else
                    {
                        oskar_correlate_point_time_smearing_cuda_d
                        (n_sources, n_stations, (const double4c*)J->data.data,
                                (const double*)sky->I.data,
                                (const double*)sky->Q.data,
                                (const double*)sky->U.data,
                                (const double*)sky->V.data,
                                (const double*)sky->rel_l.data,
                                (const double*)sky->rel_m.data,
                                (const double*)sky->rel_n.data,
                                (const double*)u->data, (const double*)v->data,
                                (const double*)telescope->station_x.data,
                                (const double*)telescope->station_y.data,
                                freq, bandwidth, time_avg, gha0,
                                telescope->dec0_rad, (double4c*)vis->data);
                    }
                }
                else
                {
                    if (sky->use_extended)
                    {
                        oskar_correlate_extended_cuda_d
                        (n_sources, n_stations, (const double4c*)J->data.data,
                                (const double*)sky->I.data,
                                (const double*)sky->Q.data,
                                (const double*)sky->U.data,
                                (const double*)sky->V.data,
                                (const double*)sky->rel_l.data,
                                (const double*)sky->rel_m.data,
                                (const double*)sky->gaussian_a.data,
                                (const double*)sky->gaussian_b.data,
                                (const double*)sky->gaussian_c.data,
                                (const double*)u->data, (const double*)v->data,
                                freq, bandwidth, (double4c*)vis->data);
                    }
                    else
                    {
                        oskar_correlate_point_cuda_d
                        (n_sources, n_stations, (const double4c*)J->data.data,
                                (const double*)sky->I.data,
                                (const double*)sky->Q.data,
                                (const double*)sky->U.data,
                                (const double*)sky->V.data,
                                (const double*)sky->rel_l.data,
                                (const double*)sky->rel_m.data,
                                (const double*)u->data, (const double*)v->data,
                                lambda_bandwidth, (double4c*)vis->data);
                    }
                }
            }
            else if (base_type == OSKAR_SINGLE)
            {
                if (time_avg > 0.0)
                {
                    if (sky->use_extended)
                    {
                        oskar_correlate_extended_time_smearing_cuda_f
                        (n_sources, n_stations, (const float4c*)J->data.data,
                                (const float*)sky->I.data,
                                (const float*)sky->Q.data,
                                (const float*)sky->U.data,
                                (const float*)sky->V.data,
                                (const float*)sky->rel_l.data,
                                (const float*)sky->rel_m.data,
                                (const float*)sky->rel_n.data,
                                (const float*)sky->gaussian_a.data,
                                (const float*)sky->gaussian_b.data,
                                (const float*)sky->gaussian_c.data,
                                (const float*)u->data, (const float*)v->data,
                                (const float*)telescope->station_x.data,
                                (const float*)telescope->station_y.data,
                                freq, bandwidth, time_avg, gha0,
                                telescope->dec0_rad, (float4c*)vis->data);
                    }
                    else
                    {
                        oskar_correlate_point_time_smearing_cuda_f
                        (n_sources, n_stations, (const float4c*)J->data.data,
                                (const float*)sky->I.data,
                                (const float*)sky->Q.data,
                                (const float*)sky->U.data,
                                (const float*)sky->V.data,
                                (const float*)sky->rel_l.data,
                                (const float*)sky->rel_m.data,
                                (const float*)sky->rel_n.data,
                                (const float*)u->data, (const float*)v->data,
                                (const float*)telescope->station_x.data,
                                (const float*)telescope->station_y.data,
                                freq, bandwidth, time_avg, gha0,
                                telescope->dec0_rad, (float4c*)vis->data);
                    }
                }
                else
                {
                    if (sky->use_extended)
                    {
                        oskar_correlate_extended_cuda_f
                        (n_sources, n_stations, (const float4c*)J->data.data,
                                (const float*)sky->I.data,
                                (const float*)sky->Q.data,
                                (const float*)sky->U.data,
                                (const float*)sky->V.data,
                                (const float*)sky->rel_l.data,
                                (const float*)sky->rel_m.data,
                                (const float*)sky->gaussian_a.data,
                                (const float*)sky->gaussian_b.data,
                                (const float*)sky->gaussian_c.data,
                                (const float*)u->data, (const float*)v->data,
                                freq, bandwidth, (float4c*)vis->data);
                    }
                    else
                    {
                        oskar_correlate_point_cuda_f
                        (n_sources, n_stations, (const float4c*)J->data.data,
                                (const float*)sky->I.data,
                                (const float*)sky->Q.data,
                                (const float*)sky->U.data,
                                (const float*)sky->V.data,
                                (const float*)sky->rel_l.data,
                                (const float*)sky->rel_m.data,
                                (const float*)u->data, (const float*)v->data,
                                lambda_bandwidth, (float4c*)vis->data);
                    }
                }
            }
        }

        /* Jones type is a scalar. */
        else
        {
            if (sky->use_extended)
            {
                *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
                return;
            }

            if (base_type == OSKAR_DOUBLE)
            {
                oskar_correlate_point_scalar_cuda_d
                (n_sources, n_stations, (const double2*)J->data.data,
                        (const double*)sky->I.data,
                        (const double*)sky->rel_l.data,
                        (const double*)sky->rel_m.data,
                        (const double*)u->data, (const double*)v->data,
                        lambda_bandwidth, (double2*)vis->data);
            }
            else if (base_type == OSKAR_SINGLE)
            {
                oskar_correlate_point_scalar_cuda_f
                (n_sources, n_stations, (const float2*)J->data.data,
                        (const float*)sky->I.data,
                        (const float*)sky->rel_l.data,
                        (const float*)sky->rel_m.data,
                        (const float*)u->data, (const float*)v->data,
                        lambda_bandwidth, (float2*)vis->data);
            }
        }

        oskar_cuda_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }

    /* Memory is on the host. */
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

#ifdef __cplusplus
}
#endif
