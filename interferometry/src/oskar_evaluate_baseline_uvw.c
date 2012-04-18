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

#include "interferometry/oskar_evaluate_baseline_uvw.h"
#include "interferometry/oskar_evaluate_baselines.h"
#include "interferometry/oskar_evaluate_station_uvw.h"
#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_init.h"
#include "sky/oskar_mjd_to_gast_fast.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_baseline_uvw(oskar_Visibilities* vis,
        const oskar_TelescopeModel* telescope, const oskar_SettingsTime* times)
{
    oskar_Mem u, v, w, work;
    oskar_Mem uu, vv, ww; /* Pointers. */
    int err, i, type, num_stations, num_baselines, num_vis_dumps;
    double obs_start_mjd_utc, dt_dump;

    /* Sanity check on inputs. */
    if (vis == NULL || telescope == NULL || times == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Get data type and size of the telescope structure. */
    type = oskar_telescope_model_type(telescope);
    num_stations = telescope->num_stations;
    num_baselines = num_stations * (num_stations - 1) / 2;

    /* Get time data. */
    num_vis_dumps     = times->num_time_steps;
    obs_start_mjd_utc = times->obs_start_mjd_utc;
    dt_dump           = times->dt_dump_days;

    /* Check that the memory is not NULL. */
    if (!vis->uu_metres.data || !vis->vv_metres.data || !vis->ww_metres.data ||
            !telescope->station_x.data || !telescope->station_y.data ||
            !telescope->station_z.data)
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check that the data dimensions are OK. */
    if (vis->uu_metres.num_elements < num_baselines * num_vis_dumps ||
            vis->vv_metres.num_elements < num_baselines * num_vis_dumps ||
            vis->ww_metres.num_elements < num_baselines * num_vis_dumps ||
            telescope->station_x.num_elements < num_stations ||
            telescope->station_y.num_elements < num_stations ||
            telescope->station_z.num_elements < num_stations)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check that the data is in the right location. */
    if (oskar_telescope_model_location(telescope) != OSKAR_LOCATION_CPU ||
            vis->uu_metres.location != OSKAR_LOCATION_CPU ||
            vis->vv_metres.location != OSKAR_LOCATION_CPU ||
            vis->ww_metres.location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Check that the data is of the right type. */
    if (vis->uu_metres.type != type ||
            vis->vv_metres.type != type ||
            vis->ww_metres.type != type)
        return OSKAR_ERR_TYPE_MISMATCH;

    /* Create a local CPU work buffer. */
    err = oskar_mem_init(&work, type, OSKAR_LOCATION_CPU, 3 * num_stations, 1);
    if (err) return err;
    err = oskar_mem_get_pointer(&u, &work, 0, num_stations);
    if (err) return err;
    err = oskar_mem_get_pointer(&v, &work, 1 * num_stations, num_stations);
    if (err) return err;
    err = oskar_mem_get_pointer(&w, &work, 2 * num_stations, num_stations);
    if (err) return err;

    /* Loop over dumps. */
    for (i = 0; i < num_vis_dumps; ++i)
    {
        double t_dump, gast;

        t_dump = obs_start_mjd_utc + i * dt_dump;
        gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

        /* Compute u,v,w coordinates of mid point. */
        err = oskar_evaluate_station_uvw(&u, &v, &w, telescope, gast);
        if (err) return err;

        /* Extract pointers to baseline u,v,w coordinates for this dump. */
        err = oskar_mem_get_pointer(&uu, &vis->uu_metres, i * num_baselines,
                num_baselines);
        if (err) return err;
        err = oskar_mem_get_pointer(&vv, &vis->vv_metres, i * num_baselines,
                num_baselines);
        if (err) return err;
        err = oskar_mem_get_pointer(&ww, &vis->ww_metres, i * num_baselines,
                num_baselines);
        if (err) return err;

        /* Compute baselines from station positions. */
        err = oskar_evaluate_baselines(&uu, &vv, &ww, &u, &v, &w);
        if (err) return err;
    }

    /* Free the work buffer. */
    oskar_mem_free(&work);

    return 0;
}

#ifdef __cplusplus
}
#endif
