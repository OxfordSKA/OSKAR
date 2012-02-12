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

#include "interferometry/oskar_evaluate_baseline_uvw.h"
#include "interferometry/oskar_evaluate_baselines.h"
#include "interferometry/oskar_evaluate_station_uvw.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "sky/oskar_mjd_to_gast_fast.h"

extern "C"
int oskar_evaluate_baseline_uvw(oskar_Visibilities* vis,
        const oskar_TelescopeModel* telescope, const oskar_SettingsTime* times)
{
    // Assert that the parameters are not NULL.
    if (vis == NULL || telescope == NULL || times == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Get data type and size of the telescope structure.
    int type = oskar_telescope_model_type(telescope);
    int num_stations = telescope->num_stations;
    int num_baselines = num_stations * (num_stations - 1) / 2;

    // Get time data.
    int num_vis_dumps        = times->num_vis_dumps;
    double obs_start_mjd_utc = times->obs_start_mjd_utc;
    double dt_dump           = times->dt_dump_days;

    // Check that the memory is not NULL.
    if (vis->uu_metres.is_null() ||
            vis->vv_metres.is_null() ||
            vis->ww_metres.is_null() ||
            telescope->station_x.is_null() ||
            telescope->station_y.is_null() ||
            telescope->station_z.is_null())
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    // Check that the data dimensions are OK.
    if (vis->uu_metres.num_elements() < num_baselines * num_vis_dumps ||
            vis->vv_metres.num_elements() < num_baselines * num_vis_dumps ||
            vis->ww_metres.num_elements() < num_baselines * num_vis_dumps ||
            telescope->station_x.num_elements() < num_stations ||
            telescope->station_y.num_elements() < num_stations ||
            telescope->station_z.num_elements() < num_stations)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    // Check that the data is in the right location.
    if (telescope->location() != OSKAR_LOCATION_CPU ||
            vis->uu_metres.location() != OSKAR_LOCATION_CPU ||
            vis->vv_metres.location() != OSKAR_LOCATION_CPU ||
            vis->ww_metres.location() != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Check that the data is of the right type.
    if (vis->uu_metres.type() != type ||
            vis->vv_metres.type() != type ||
            vis->ww_metres.type() != type)
        return OSKAR_ERR_TYPE_MISMATCH;

    // Create a local CPU work buffer.
    oskar_Mem work(type, OSKAR_LOCATION_CPU, 3 * num_stations);
    oskar_Mem u = work.get_pointer(0, num_stations);
    oskar_Mem v = work.get_pointer(1 * num_stations, num_stations);
    oskar_Mem w = work.get_pointer(2 * num_stations, num_stations);
    oskar_Mem uu, vv, ww; // Pointers.

    // Loop over dumps.
    for (int i = 0; i < num_vis_dumps; ++i)
    {
        int err = 0;
        double t_dump = obs_start_mjd_utc + i * dt_dump;
        double gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

        // Compute u,v,w coordinates of mid point.
        err = oskar_evaluate_station_uvw(&u, &v, &w, telescope, gast);
        if (err) return err;

        // Extract pointers to baseline u,v,w coordinates for this dump.
        uu = vis->uu_metres.get_pointer(i * num_baselines, num_baselines);
        vv = vis->vv_metres.get_pointer(i * num_baselines, num_baselines);
        ww = vis->ww_metres.get_pointer(i * num_baselines, num_baselines);
        if (uu.data == NULL || vv.data == NULL || ww.data == NULL)
            return OSKAR_ERR_UNKNOWN;

        // Compute baselines from station positions.
        err = oskar_evaluate_baselines(&uu, &vv, &ww, &u, &v, &w);
        if (err) return err;
    }

    return 0;
}
