/*
 * Copyright (c) 2011-2017, The University of Oxford
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

#include "math/oskar_prefix_sum.h"
#include "sky/oskar_sky.h"
#include "sky/oskar_sky_copy_source_data.h"
#include "sky/oskar_update_horizon_mask.h"

#ifdef __cplusplus
extern "C" {
#endif

static double ha0(double longitude, double ra0, double gast);

void oskar_sky_horizon_clip(oskar_Sky* out, const oskar_Sky* in,
        const oskar_Telescope* telescope, double gast,
        oskar_StationWork* work, int* status)
{
    int i, num_stations, location, num_in;
    oskar_Mem *horizon_mask, *source_indices;
    double ra0, dec0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get pointers to work arrays. */
    horizon_mask = oskar_station_work_horizon_mask(work);
    source_indices = oskar_station_work_source_indices(work);

    /* Check that the types match. */
    if (oskar_sky_precision(in) != oskar_sky_precision(out))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check that the locations match. */
    location = oskar_sky_mem_location(out);
    if (oskar_sky_mem_location(in) != location ||
            oskar_mem_location(horizon_mask) != location ||
            oskar_mem_location(source_indices) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Get remaining properties of input sky model. */
    num_in = oskar_sky_num_sources(in);
    ra0 = oskar_sky_reference_ra_rad(in);
    dec0 = oskar_sky_reference_dec_rad(in);

    /* Resize the output sky model if necessary. */
    if (oskar_sky_capacity(out) < num_in)
        oskar_sky_resize(out, num_in, status);

    /* Resize the work buffers if necessary. */
    if ((int)oskar_mem_length(horizon_mask) < num_in)
        oskar_mem_realloc(horizon_mask, num_in, status);
    if ((int)oskar_mem_length(source_indices) < num_in)
        oskar_mem_realloc(source_indices, num_in, status);

    /* Create the horizon mask. */
    oskar_mem_clear_contents(horizon_mask, status);
    num_stations = oskar_telescope_num_stations(telescope);
    for (i = 0; i < num_stations; ++i)
    {
        const oskar_Station* s = oskar_telescope_station_const(telescope, i);
        oskar_update_horizon_mask(num_in, oskar_sky_l_const(in),
                oskar_sky_m_const(in), oskar_sky_n_const(in),
                ha0(oskar_station_lon_rad(s), ra0, gast), dec0,
                oskar_station_lat_rad(s), horizon_mask, status);
    }

    /* Apply exclusive prefix sum to mask to get source output indices. */
    if (location != OSKAR_CPU)
        oskar_prefix_sum(num_in, horizon_mask, source_indices, 0, 1, status);

    /* Copy sources above horizon. */
    oskar_sky_copy_source_data(in, horizon_mask, source_indices, out, status);
}

static double ha0(double longitude, double ra0, double gast)
{
    return (gast + longitude) - ra0;
}

#ifdef __cplusplus
}
#endif
