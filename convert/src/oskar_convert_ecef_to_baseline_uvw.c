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

#include <oskar_convert_ecef_to_baseline_uvw.h>
#include <oskar_convert_station_uvw_to_baseline_uvw.h>
#include <oskar_convert_ecef_to_station_uvw.h>
#include <oskar_convert_mjd_to_gast_fast.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_ecef_to_baseline_uvw(int num_stations, const oskar_Mem* x,
        const oskar_Mem* y, const oskar_Mem* z, double ra0_rad,
        double dec0_rad, int num_times, double time_ref_mjd_utc,
        double time_inc_days, int start_time_index, oskar_Mem* uu,
        oskar_Mem* vv, oskar_Mem* ww, oskar_Mem* work, int* status)
{
    oskar_Mem *u, *v, *w, *uu_dump, *vv_dump, *ww_dump; /* Aliases. */
    int i, type, location, num_baselines;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that the data dimensions are OK. */
    num_baselines = num_stations * (num_stations - 1) / 2;
    if ((int)oskar_mem_length(uu) < num_baselines * num_times ||
            (int)oskar_mem_length(vv) < num_baselines * num_times ||
            (int)oskar_mem_length(ww) < num_baselines * num_times ||
            (int)oskar_mem_length(x) < num_stations ||
            (int)oskar_mem_length(y) < num_stations ||
            (int)oskar_mem_length(z) < num_stations)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if ((int)oskar_mem_length(work) < 3 * num_stations)
        oskar_mem_realloc(work, 3 * num_stations, status);
    if (*status) return;

    /* Check that the data are of the right type. */
    type = oskar_mem_type(x);
    if (oskar_mem_type(uu) != type || oskar_mem_type(vv) != type ||
            oskar_mem_type(ww) != type || oskar_mem_type(y) != type ||
            oskar_mem_type(z) != type || oskar_mem_type(work) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check that the data are in the right location. */
    location = oskar_mem_location(x);
    if (oskar_mem_location(y) != location ||
            oskar_mem_location(z) != location ||
            oskar_mem_location(uu) != location ||
            oskar_mem_location(vv) != location ||
            oskar_mem_location(ww) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Create pointers from work buffer. */
    u = oskar_mem_create_alias(work, 0, num_stations, status);
    v = oskar_mem_create_alias(work, 1 * num_stations, num_stations, status);
    w = oskar_mem_create_alias(work, 2 * num_stations, num_stations, status);

    /* Create pointers to baseline u,v,w coordinates for dump. */
    uu_dump = oskar_mem_create_alias(0, 0, 0, status);
    vv_dump = oskar_mem_create_alias(0, 0, 0, status);
    ww_dump = oskar_mem_create_alias(0, 0, 0, status);

    /* Loop over times. */
    for (i = 0; i < num_times; ++i)
    {
        double t_dump, gast;

        t_dump = time_ref_mjd_utc +
                time_inc_days * ((i + 0.5) + start_time_index);
        gast = oskar_convert_mjd_to_gast_fast(t_dump);

        /* Compute u,v,w coordinates of mid point. */
        oskar_convert_ecef_to_station_uvw(num_stations, x, y, z,
                ra0_rad, dec0_rad, gast, u, v, w, status);

        /* Extract pointers to baseline u,v,w coordinates for this dump. */
        oskar_mem_set_alias(uu_dump, uu, i * num_baselines,
                num_baselines, status);
        oskar_mem_set_alias(vv_dump, vv, i * num_baselines,
                num_baselines, status);
        oskar_mem_set_alias(ww_dump, ww, i * num_baselines,
                num_baselines, status);

        /* Compute baselines from station positions. */
        oskar_convert_station_uvw_to_baseline_uvw(u, v, w,
                uu_dump, vv_dump, ww_dump, status);
    }

    /* Free handles to aliased memory. */
    oskar_mem_free(u, status);
    oskar_mem_free(v, status);
    oskar_mem_free(w, status);
    oskar_mem_free(uu_dump, status);
    oskar_mem_free(vv_dump, status);
    oskar_mem_free(ww_dump, status);
}

#ifdef __cplusplus
}
#endif
