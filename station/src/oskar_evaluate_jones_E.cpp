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

#include "oskar_global.h"
#include <cuda_runtime_api.h>

#include "station/oskar_evaluate_jones_E.h"

#include "sky/oskar_SkyModel.h"
#include "sky/oskar_ra_dec_to_hor_lmn.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "math/oskar_Jones.h"
#include "math/oskar_jones_get_station_pointer.h"
#include "station/oskar_evaluate_station_beam.h"
#include "station/oskar_evaluate_beam_horizontal_lmn.h"
#include "station/oskar_evaluate_source_horizontal_lmn.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_Work.h"
#include <cstdio>

#ifdef __cplusplus
extern "C"
#endif
int oskar_evaluate_jones_E(oskar_Jones* E, const oskar_SkyModel* sky,
        const oskar_TelescopeModel* telescope, const double gast,
        oskar_Work* work, oskar_Device_curand_state* curand_state)
{
    int error = 0;

    // Consistency and validation checks on input arguments.
    if (E == NULL || sky == NULL || telescope == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (telescope->num_stations == 0)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    if (telescope->station == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (work->real.type() != OSKAR_DOUBLE && work->real.type() != OSKAR_SINGLE)
        return OSKAR_ERR_BAD_DATA_TYPE;

    if (E->location() != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Resize the work buffer if needed.
    if (work->real.num_elements() < 3 * sky->num_sources)
    {
        error = work->real.resize(3 * sky->num_sources);
        if (error) return error;
    }


    // Get pointers to work arrays.
    oskar_Mem* weights = &work->complex;
    double beam_l, beam_m, beam_n;
    oskar_Mem hor_l, hor_m, hor_n;
    error = oskar_mem_get_pointer(&hor_l, &work->real, 0, sky->num_sources);
    if (error) return error;
    error = oskar_mem_get_pointer(&hor_m, &work->real, sky->num_sources,
            sky->num_sources);
    if (error) return error;
    error = oskar_mem_get_pointer(&hor_n, &work->real, 2 * sky->num_sources,
            sky->num_sources);
    if (error) return error;


    // Evaluate the station beam for each station for each source position.
    if (telescope->identical_stations && telescope->use_common_sky)
    {
        // Evaluate horizontal l,m,n once and the station beam for
        // station 0 and copy the result into the data for other stations
        // in E.
        oskar_StationModel* station0 = &telescope->station[0];

        // Evaluate the horizontal l,m,m coordinates of the beam phase centre
        // and sources.
        error = oskar_evaluate_beam_horizontal_lmn(&beam_l, &beam_m, &beam_n,
                station0, gast);
        if (error) return error;
        error = oskar_evaluate_source_horizontal_lmn(&hor_l, &hor_m, &hor_n,
                &sky->RA, &sky->Dec, station0, gast);
        if (error) return error;

        // Evaluate the station beam.
        oskar_Mem E0; // Pointer to the row of E for station 0.
        error = oskar_jones_get_station_pointer(&E0, E, 0);
        if (error) return error;
        error = oskar_evaluate_station_beam(&E0, station0, beam_l, beam_m,
                &hor_l, &hor_m, &hor_n, weights, curand_state);
        if (error) return error;

        // Copy E for station 0 into memory for other stations.
        oskar_Mem E_station;
        size_t mem_size = E->num_sources() * oskar_mem_element_size(E->type());
        for (int i = 1; i < telescope->num_stations; ++i)
        {
            error = oskar_jones_get_station_pointer(&E_station, E, i);
            if (error) return error;
            cudaMemcpy(E_station.data, E0.data, mem_size, cudaMemcpyDeviceToDevice);
        }
    }
    else
    {
        if (telescope->use_common_sky)
        {
            // Evaluate horizontal l,m,n once and use it for evaluating
            // the station beam for each station.
            oskar_StationModel* station0 = &telescope->station[0];
            error = oskar_evaluate_beam_horizontal_lmn(&beam_l, &beam_m,
                    &beam_n, station0, gast);
            if (error) return error;
            error = oskar_evaluate_source_horizontal_lmn(&hor_l, &hor_m, &hor_n,
                    &sky->RA, &sky->Dec, station0, gast);
            if (error) return error;

            // loop over stations to evaluate E.
            oskar_Mem E_station;
            for (int i = 0; i < telescope->num_stations; ++i)
            {
                oskar_StationModel* station = &telescope->station[i];
                error = oskar_jones_get_station_pointer(&E_station, E, i);
                if (error) return error;
                error = oskar_evaluate_station_beam(&E_station, station, beam_l,
                        beam_m, &hor_l, &hor_m, &hor_n, weights, curand_state);
                if (error) return error;
            }
        }
        else
        {
            // Evaluate horizontal l,m,n and the station beam for each station.
            // loop over stations to evaluate E.
            oskar_Mem E_station;
            for (int i = 0; i < telescope->num_stations; ++i)
            {
                oskar_StationModel* station = &telescope->station[i];
                error = oskar_evaluate_beam_horizontal_lmn(&beam_l, &beam_m,
                        &beam_n, station, gast);
                if (error) return error;
                error = oskar_evaluate_source_horizontal_lmn(&hor_l, &hor_m,
                        &hor_n, &sky->RA, &sky->Dec, station, gast);
                if (error) return error;
                error = oskar_jones_get_station_pointer(&E_station, E, i);
                if (error) return error;
                error = oskar_evaluate_station_beam(&E_station, station,
                        beam_l, beam_m, &hor_l, &hor_m,  &hor_n, weights,
                        curand_state);
                if (error) return error;
            }
        }
    }
    return 0;
}
