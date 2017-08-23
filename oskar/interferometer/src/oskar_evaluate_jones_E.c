/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include "interferometer/oskar_evaluate_jones_E.h"
#include "interferometer/oskar_jones_get_station_pointer.h"
#include "telescope/station/oskar_evaluate_station_beam.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_jones_E(oskar_Jones* E, int num_points, int coord_type,
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z, const oskar_Telescope* tel,
        double gast, double frequency_hz, oskar_StationWork* work,
        int time_index, int* status)
{
    int i, num_stations;
    oskar_Mem *E_st;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check number of stations. */
    num_stations = oskar_telescope_num_stations(tel);
    if (num_stations == 0)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Evaluate the station beams. */
    E_st = oskar_mem_create_alias(0, 0, 0, status);
    if (oskar_telescope_allow_station_beam_duplication(tel) &&
            oskar_telescope_identical_stations(tel))
    {
        /* Identical stations. */
        oskar_Mem *E0; /* Pointer to row of E for station 0. */
        const oskar_Station* station0;

        /* Evaluate the beam pattern for station 0 */
        E0 = oskar_mem_create_alias(0, 0, 0, status);
        station0 = oskar_telescope_station_const(tel, 0);
        oskar_jones_get_station_pointer(E0, E, 0, status);

        oskar_evaluate_station_beam(E0, num_points, coord_type, x, y, z,
                oskar_telescope_phase_centre_ra_rad(tel),
                oskar_telescope_phase_centre_dec_rad(tel),
                station0, work, time_index, frequency_hz, gast, status);

        /* Copy E for station 0 into memory for other stations. */
        for (i = 1; i < num_stations; ++i)
        {
            oskar_jones_get_station_pointer(E_st, E, i, status);
            oskar_mem_copy_contents(E_st, E0, 0, 0,
                    oskar_mem_length(E0), status);
        }
        oskar_mem_free(E0, status);
    }
    else
    {
        /* Different stations. */
        for (i = 0; i < num_stations; ++i)
        {
            const oskar_Station* station;
            station = oskar_telescope_station_const(tel, i);
            oskar_jones_get_station_pointer(E_st, E, i, status);
            oskar_evaluate_station_beam(E_st, num_points, coord_type, x, y, z,
                    oskar_telescope_phase_centre_ra_rad(tel),
                    oskar_telescope_phase_centre_dec_rad(tel),
                    station, work, time_index, frequency_hz, gast, status);
        }
    }
    oskar_mem_free(E_st, status);
}

#ifdef __cplusplus
}
#endif
