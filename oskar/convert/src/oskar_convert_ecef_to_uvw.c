/*
 * Copyright (c) 2013-2019, The University of Oxford
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

#include "convert/oskar_convert_ecef_to_uvw.h"
#include "convert/oskar_convert_station_uvw_to_baseline_uvw.h"
#include "convert/oskar_convert_ecef_to_station_uvw.h"
#include "convert/oskar_convert_mjd_to_gast_fast.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_ecef_to_uvw(int num_stations,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double ra0_rad, double dec0_rad, int num_times,
        double time_ref_mjd_utc, double time_inc_days, int start_time_index,
        oskar_Mem* u, oskar_Mem* v, oskar_Mem* w,
        oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww, int* status)
{
    int i;
    if (*status) return;
    const int num_baselines = num_stations * (num_stations - 1) / 2;
    const int total_baselines = num_baselines * num_times;
    const int total_stations = num_stations * num_times;
    oskar_mem_ensure(u, total_stations, status);
    oskar_mem_ensure(v, total_stations, status);
    oskar_mem_ensure(w, total_stations, status);
    if (uu) oskar_mem_ensure(uu, total_baselines, status);
    if (vv) oskar_mem_ensure(vv, total_baselines, status);
    if (ww) oskar_mem_ensure(ww, total_baselines, status);
    if (*status) return;
    for (i = 0; i < num_times; ++i)
    {
        const double t_dump = time_ref_mjd_utc +
                time_inc_days * ((i + 0.5) + start_time_index);
        const double gast = oskar_convert_mjd_to_gast_fast(t_dump);
        oskar_convert_ecef_to_station_uvw(num_stations, x, y, z,
                ra0_rad, dec0_rad, gast, i * num_stations, u, v, w, status);
        if (uu && vv && ww)
            oskar_convert_station_uvw_to_baseline_uvw(num_stations,
                    i * num_stations, u, v, w,
                    i * num_baselines, uu, vv, ww, status);
    }
}

#ifdef __cplusplus
}
#endif
