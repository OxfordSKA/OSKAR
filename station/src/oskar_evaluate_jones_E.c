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

#include "interferometry/oskar_telescope_model_location.h"
#include "math/oskar_jones_get_station_pointer.h"
#include "sky/oskar_sky_model_location.h"
#include "station/oskar_evaluate_beam_horizontal_lmn.h"
#include "station/oskar_evaluate_jones_E.h"
#include "station/oskar_evaluate_source_horizontal_lmn.h"
#include "station/oskar_evaluate_station_beam.h"
#include "utility/oskar_mem_insert.h"

#ifdef __cplusplus
extern "C" {
#endif

static void evaluate_E_common_sky_identical_stations(oskar_Jones* E,
        const oskar_SkyModel* sky, const oskar_TelescopeModel* telescope,
        double gast, oskar_WorkStationBeam* work,
        oskar_CurandState* curand_state, int* status);
static void evaluate_E_common_sky_different_stations(oskar_Jones* E,
        const oskar_SkyModel* sky, const oskar_TelescopeModel* telescope,
        double gast, oskar_WorkStationBeam* work,
        oskar_CurandState* curand_state, int* status);
static void evaluate_E_different_sky(oskar_Jones* E,
        const oskar_SkyModel* sky, const oskar_TelescopeModel* telescope,
        double gast, oskar_WorkStationBeam* work,
        oskar_CurandState* curand_state, int* status);

void oskar_evaluate_jones_E(oskar_Jones* E, const oskar_SkyModel* sky,
        const oskar_TelescopeModel* telescope, double gast,
        oskar_WorkStationBeam* work, oskar_CurandState* curand_state,
        int* status)
{
    /* Check all inputs. */
    if (!E || !sky || !telescope || !work || !curand_state || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    if (!telescope->station || telescope->num_stations == 0)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Evaluate the station beam for each station at each source position. */
    /* A number of optimisations are possible, so switch on these. */
    if (telescope->use_common_sky)
    {
        /* Optimisation 1: only evaluate one beam and copy. */
        if (telescope->identical_stations)
        {
            evaluate_E_common_sky_identical_stations(E, sky, telescope,
                    gast, work, curand_state, status);
        }

        /* Optimisation 2: share sky coordinates between beam evaluations */
        else /* (!telescope->identical_stations) */
        {
            evaluate_E_common_sky_different_stations(E, sky, telescope,
                    gast, work, curand_state, status);
        }
    }

    /* No optimisation possible.
     * Evaluate the beam per station using different sky coordinates. */
    else /* (!telescope->use_common_sky) */
    {
        evaluate_E_different_sky(E, sky, telescope, gast,
                work, curand_state, status);
    }
}

/*
 * Optimisation 1:
 * With a common sky (horizon) and identical stations, all station beams
 * will be the same. This function evaluates the beam once (for station 0) and
 * then copies it into the other station indices in the Jones matrix structure.
 */
static void evaluate_E_common_sky_identical_stations(oskar_Jones* E,
        const oskar_SkyModel* sky, const oskar_TelescopeModel* telescope,
        double gast, oskar_WorkStationBeam* work,
        oskar_CurandState* curand_state, int* status)
{
    double beam_l, beam_m, beam_n;
    int i;

    /* Evaluate source horizontal l,m,n once, and copy the station beam for
     * station 0 into the data for other stations in E. */
    oskar_Mem E0; /* Pointer to the row of E for station 0. */
    oskar_StationModel* station0;
    station0 = &telescope->station[0];

    /* Check if safe to proceed. */
    if (*status) return;

    /* Evaluate the horizontal l,m,n coordinates of the beam phase centre
     * and sources. */
    oskar_evaluate_beam_horizontal_lmn(&beam_l, &beam_m, &beam_n,
            station0, gast, status);
    oskar_evaluate_source_horizontal_lmn(sky->num_sources,
            &work->hor_x, &work->hor_y, &work->hor_z, &sky->RA, &sky->Dec,
            station0, gast, status);
    oskar_jones_get_station_pointer(&E0, E, 0, status);

    if (station0->station_type == OSKAR_STATION_TYPE_AA)
    {
        oskar_evaluate_station_beam(&E0, station0, beam_l, beam_m, beam_n,
                sky->num_sources, OSKAR_BEAM_COORDS_HORIZONTAL,
                &work->hor_x, &work->hor_y, &work->hor_z, &work->hor_z,
                work, curand_state, status);
    }
    else if (station0->station_type == OSKAR_STATION_TYPE_GAUSSIAN_BEAM)
    {
        oskar_evaluate_station_beam(&E0, station0, beam_l, beam_m, beam_n,
                sky->num_sources, OSKAR_BEAM_COORDS_PHASE_CENTRE,
                &sky->rel_l, &sky->rel_m, &sky->rel_n, &work->hor_z,
                work, curand_state, status);
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
    }

    /* Copy E for station 0 into memory for other stations. */
    for (i = 1; i < telescope->num_stations; ++i)
    {
        oskar_Mem E_station;
        oskar_jones_get_station_pointer(&E_station, E, i, status);
        oskar_mem_insert(&E_station, &E0, 0, status);
    }
}

/*
 * Optimisation 2:
 * With a common sky (horizon) but different stations, while all station
 * beams must be evaluated separately, the beam coordinates can be shared
 * and only evaluated once.
 */
static void evaluate_E_common_sky_different_stations(oskar_Jones* E,
        const oskar_SkyModel* sky, const oskar_TelescopeModel* telescope,
        double gast, oskar_WorkStationBeam* work,
        oskar_CurandState* curand_state, int* status)
{
    double beam_l, beam_m, beam_n;
    int i;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Evaluate horizontal l,m,n once and use them to evaluate
     * the station beam for each station. */
    oskar_evaluate_beam_horizontal_lmn(&beam_l, &beam_m,
            &beam_n, &telescope->station[0], gast, status);
    oskar_evaluate_source_horizontal_lmn(sky->num_sources,
            &work->hor_x, &work->hor_y, &work->hor_z, &sky->RA, &sky->Dec,
            &telescope->station[0], gast, status);

    for (i = 0; i < telescope->num_stations; ++i)
    {
        oskar_Mem E_station;
        oskar_StationModel* station = &telescope->station[i];
        oskar_jones_get_station_pointer(&E_station, E, i, status);

        if (station->station_type == OSKAR_STATION_TYPE_AA)
        {
            oskar_evaluate_station_beam(&E_station, station, beam_l, beam_m,
                    beam_n, sky->num_sources,
                    OSKAR_BEAM_COORDS_HORIZONTAL, &work->hor_x,
                    &work->hor_y, &work->hor_z, &work->hor_z, work,
                    curand_state, status);
        }
        else if (station->station_type == OSKAR_STATION_TYPE_GAUSSIAN_BEAM)
        {
            oskar_evaluate_station_beam(&E_station, station, beam_l, beam_m,
                    beam_n, sky->num_sources,
                    OSKAR_BEAM_COORDS_PHASE_CENTRE, &sky->rel_l,
                    &sky->rel_m, &sky->rel_n, &work->hor_z, work,
                    curand_state, status);
        }
        else
        {
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        }
    }
}

/*
 * No optimisation:
 * Full E evaluation where the beam evaluated for each station, each having a
 * different set of sky coordinates.
 */
static void evaluate_E_different_sky(oskar_Jones* E,
        const oskar_SkyModel* sky, const oskar_TelescopeModel* telescope,
        double gast, oskar_WorkStationBeam* work,
        oskar_CurandState* curand_state, int* status)
{
    double beam_l, beam_m, beam_n;
    int i;

    /* Check if safe to proceed. */
    if (*status) return;

    for (i = 0; i < telescope->num_stations; ++i)
    {
        oskar_Mem E_station;
        oskar_StationModel* station = &telescope->station[i];

        oskar_evaluate_beam_horizontal_lmn(&beam_l, &beam_m,
                &beam_n, station, gast, status);
        oskar_evaluate_source_horizontal_lmn(sky->num_sources,
                &work->hor_x, &work->hor_y, &work->hor_z, &sky->RA, &sky->Dec,
                station, gast, status);
        oskar_jones_get_station_pointer(&E_station, E, i, status);

        if (station->station_type == OSKAR_STATION_TYPE_AA)
        {
            oskar_evaluate_station_beam(&E_station, station, beam_l,
                    beam_m, beam_n, sky->num_sources,
                    OSKAR_BEAM_COORDS_HORIZONTAL,
                    &work->hor_x, &work->hor_y, &work->hor_z, &work->hor_z,
                    work, curand_state, status);
        }
        else if (station->station_type == OSKAR_STATION_TYPE_GAUSSIAN_BEAM)
        {
            oskar_evaluate_station_beam(&E_station, station, beam_l, beam_m,
                    beam_n, sky->num_sources,
                    OSKAR_BEAM_COORDS_PHASE_CENTRE, &sky->rel_l,
                    &sky->rel_m, &sky->rel_n, &work->hor_z, work,
                    curand_state, status);
        }
        else
        {
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        }
    }
}

#ifdef __cplusplus
}
#endif
