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

#include "interferometry/oskar_evaluate_baselines.h"
#include "interferometry/oskar_evaluate_uvw_baseline.h"
#include "interferometry/oskar_evaluate_uvw_station.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_get_pointer.h"
#include "sky/oskar_mjd_to_gast_fast.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_uvw_baseline(oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww,
        int num_stations, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, double ra0_rad, double dec0_rad, int num_dumps,
        double start_mjd_utc, double dt_dump_days, oskar_Mem* work,
        int* status)
{
    oskar_Mem u, v, w, uu_dump, vv_dump, ww_dump; /* Pointers. */
    int i, type, location, num_baselines;

    /* Check all inputs. */
    if (!uu || !vv || !ww || !x || !y || !z || !work || !status)
    {
        if (status) *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get input data type and number of baselines. */
    type = x->type;
    num_baselines = num_stations * (num_stations - 1) / 2;

    /* Check that the memory is not NULL. */
    if (!uu->data || !vv->data || !ww->data || !x->data || !y->data ||
            !z->data || !work->data)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check that the data dimensions are OK. */
    if (uu->num_elements < num_baselines * num_dumps ||
            vv->num_elements < num_baselines * num_dumps ||
            ww->num_elements < num_baselines * num_dumps ||
            x->num_elements < num_stations ||
            y->num_elements < num_stations ||
            z->num_elements < num_stations ||
            work->num_elements < 3 * num_stations)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check that the data is of the right type. */
    if (uu->type != type || vv->type != type || ww->type != type ||
            y->type != type || z->type != type || work->type != type)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Check that the data is in the right location. */
    location = x->location;
    if (y->location != location || z->location != location ||
            uu->location != location || vv->location != location ||
            ww->location != location)
        *status = OSKAR_ERR_BAD_LOCATION;

    /* Get pointers from work buffer. */
    oskar_mem_get_pointer(&u, work, 0, num_stations, status);
    oskar_mem_get_pointer(&v, work, 1 * num_stations, num_stations, status);
    oskar_mem_get_pointer(&w, work, 2 * num_stations, num_stations, status);

    /* Loop over dumps. */
    for (i = 0; i < num_dumps; ++i)
    {
        double t_dump, gast;

        t_dump = start_mjd_utc + i * dt_dump_days;
        gast = oskar_mjd_to_gast_fast(t_dump + dt_dump_days / 2.0);

        /* Compute u,v,w coordinates of mid point. */
        oskar_evaluate_uvw_station(&u, &v, &w, num_stations, x, y, z,
                ra0_rad, dec0_rad, gast, status);

        /* Extract pointers to baseline u,v,w coordinates for this dump. */
        oskar_mem_get_pointer(&uu_dump, uu, i * num_baselines,
                num_baselines, status);
        oskar_mem_get_pointer(&vv_dump, vv, i * num_baselines,
                num_baselines, status);
        oskar_mem_get_pointer(&ww_dump, ww, i * num_baselines,
                num_baselines, status);

        /* Compute baselines from station positions. */
        oskar_evaluate_baselines(&uu_dump, &vv_dump, &ww_dump,
                &u, &v, &w, status);
    }
}

#ifdef __cplusplus
}
#endif
