/*
 * Copyright (c) 2013, The University of Oxford
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

#include "sky/oskar_evaluate_jones_Z.h"

#include "station/oskar_evaluate_source_horizontal_lmn.h"
#include "math/oskar_jones_get_station_pointer.h"
#include "interferometry/oskar_offset_geocentric_cartesian_to_geocentric_cartesian.h"
#include "utility/oskar_vector_types.h"
#include "station/oskar_evaluate_pierce_points.h"
#include "utility/oskar_mem_add.h"
#include "sky/oskar_evaluate_mim_tid_tec.h"

#include "math.h"

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_evaluate_TEC(oskar_WorkJonesZ* work, int num_pp,
        const oskar_SettingsIonosphere* settings,
        double gast, int* status);

void oskar_evaluate_jones_Z(oskar_Jones* Z, const oskar_SkyModel* sky,
        const oskar_TelescopeModel* telescope, double gast,
        const oskar_SettingsIonosphere* settings, oskar_WorkJonesZ* work,
        int* status)
{
    int i, j;
    /* Station position in ECEF frame */
    double station_x, station_y, station_z;
    double2* Z_;

    /* Check all inputs. */
    if (!Z || !sky || !telescope || !settings || !work || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    if (!telescope->station || telescope->num_stations == 0)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /*
     * FIXME below is a --DOUBLE-- precision version ONLY
     * FIXME elevation mask implementation, elevation evaluation stability.
     * FIXME GPU vs CPU
     */

    /* Evaluate the Ionospheric phase screen for each station at each
     * source pierce point. */
    for (i = 0; i < telescope->num_stations; ++i)
    {
        oskar_Mem Z_station;
        oskar_StationModel* station = &telescope->station[i];

        /* Evaluate horizontal x,y,z source positions (for which to evaluate
         * pierce points) */
        oskar_evaluate_source_horizontal_lmn(sky->num_sources, &work->hor_x,
                &work->hor_y, &work->hor_z, &sky->RA, &sky->Dec,
                station, gast, status);

        /* Obtain station coordinates in the ECEF frame. */
        oskar_offset_geocentric_cartesian_to_geocentric_cartesian(1,
                &((double*)telescope->station_x.data)[i],
                &((double*)telescope->station_y.data)[i],
                &((double*)telescope->station_z.data)[i],
                station->longitude_rad, station->latitude_rad,
                station->altitude_m, &station_x, &station_y, &station_z);

        /* Obtain the pierce points */
        oskar_evaluate_pierce_points(&work->pp_lon, &work->pp_lat,
                &work->pp_rel_path, station->latitude_rad,
                station->latitude_rad, station->altitude_m, station_x,
                station_y, station_z, settings->TID[j].height_km * 1000.,
                sky->num_sources, &work->hor_x, &work->hor_y, &work->hor_z,
                status);

        /* Evaluate TEC values for the pierce points */
        oskar_evaluate_TEC(work, sky->num_sources, settings, gast, status);

        /* Get a pointer to the Jones matrices for the station */
        oskar_jones_get_station_pointer(&Z_station, Z, i, status);

        /* Z Jones = scalar Jones matrix */
        Z_ = (double2*)Z_station.data;

        /* Populate the Jones matrix with ionospheric phase */
        for (j = 0; j < sky->num_sources; ++j)
        {
            double arg = telescope->wavelength_metres * 25. *
                    ((double*)work->total_TEC.data)[j];

            /* Initialise as an unit scalar Z = (1 + 0i) i.e. no phase change */
            Z_[j].x = 1.0;
            Z_[j].y = 0.0;

            /* If the pierce point is below the minimum specified elevation
             * don't evaluate a phase */
            if (asin(((double*)work->hor_z.data)[j]) < settings->min_elevation)
                continue;

            /* Z phase == exp(i * lambda * 25 * tec) */
            Z_[j].x = cos(arg);
            Z_[j].y = sin(arg);
        }
    }
}


/* Evaluate the TEC value for each pierce point - note: at the moment this is
 * just the accumulation of one or more TID screens.
 * TODO convert this to a stand-alone function.
 */
static void oskar_evaluate_TEC(oskar_WorkJonesZ* work, int num_pp,
        const oskar_SettingsIonosphere* settings,
        double gast, int* status)
{
    int i;

    /* Loop over TID screens to evaluate TEC values */
    for (i = 0; i < settings->num_TID_screens; ++i)
    {
        /* Evaluate TEC values for the screen */
        oskar_evaluate_tid_mim(&work->screen_TEC, num_pp, &work->pp_lon,
                &work->pp_lat, &work->pp_rel_path, settings->TEC0,
                &settings->TID[i], gast);

        /* Accumulate into total TEC */
        /* FIXME addition is not physical for more than one TEC screen in the
         * way TIDs are currently evaluated as TEC0 is added into both screens.
         */
        oskar_mem_add(&work->total_TEC, &work->total_TEC, &work->screen_TEC,
                status);
    }
}

#ifdef __cplusplus
}
#endif
