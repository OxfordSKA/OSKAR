/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <oskar_evaluate_jones_E.h>

#include <oskar_jones_get_station_pointer.h>
#include <oskar_evaluate_station_beam_pattern.h>

#ifdef __cplusplus
extern "C" {
#endif

static void evaluate_E_common_sky_identical_stations(oskar_Jones* E,
        const oskar_Sky* sky, const oskar_Telescope* telescope,
        double gast, double frequency_hz, oskar_StationWork* work,
        oskar_RandomState* rand_state, int* status);
static void evaluate_E_different_sky(oskar_Jones* E,
        const oskar_Sky* sky, const oskar_Telescope* telescope,
        double gast, double frequency_hz, oskar_StationWork* work,
        oskar_RandomState* rand_state, int* status);

void oskar_evaluate_jones_E(oskar_Jones* E, const oskar_Sky* sky,
        const oskar_Telescope* telescope, double gast, double frequency_hz,
        oskar_StationWork* work, oskar_RandomState* random_state, int* status)
{
    /* Check all inputs. */
    if (!E || !sky || !telescope || !work || !random_state || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    if (oskar_telescope_num_stations(telescope) == 0)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Evaluate the station beams */
    if (oskar_telescope_common_horizon(telescope) &&
            oskar_telescope_identical_stations(telescope))
    {
        evaluate_E_common_sky_identical_stations(E, sky, telescope, gast,
                frequency_hz, work, random_state, status);
    }
    else
    {
        evaluate_E_different_sky(E, sky, telescope, gast, frequency_hz, work,
                random_state, status);
    }
}

/*
 * Optimisation:
 * With a common sky (horizon) and identical stations, all station beams
 * will be the same. This function evaluates the beam once (for station 0) and
 * then copies it into the other station indices in the Jones matrix structure.
 */
static void evaluate_E_common_sky_identical_stations(oskar_Jones* E,
        const oskar_Sky* sky, const oskar_Telescope* telescope,
        double gast, double frequency_hz, oskar_StationWork* work,
        oskar_RandomState* rand_state, int* status)
{
    int i, num_sources;
    oskar_Mem *E0, *E_station; /* Pointer to rows of E for stations 0 and n. */
    const oskar_Station* station0;
    const oskar_Mem *l, *m, *n;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Evaluate the beam pattern for station 0 */
    E0 = oskar_mem_create_alias(0, 0, 0, status);
    E_station = oskar_mem_create_alias(0, 0, 0, status);
    station0 = oskar_telescope_station_const(telescope, 0);
    num_sources = oskar_sky_num_sources(sky);
    oskar_jones_get_station_pointer(E0, E, 0, status);
    l = oskar_sky_l_const(sky);
    m = oskar_sky_m_const(sky);
    n = oskar_sky_n_const(sky);
    oskar_evaluate_station_beam_pattern_relative_directions(E0, num_sources,
            l, m, n, station0, work, rand_state, frequency_hz, gast, status);

    /* Copy E for station 0 into memory for other stations. */
    for (i = 1; i < oskar_telescope_num_stations(telescope); ++i)
    {
        oskar_jones_get_station_pointer(E_station, E, i, status);
        oskar_mem_insert(E_station, E0, 0, status);
    }
    oskar_mem_free(E0, status);
    oskar_mem_free(E_station, status);
}

/*
 * No optimisation:
 * Full E evaluation where the beam evaluated for each station, each having a
 * different set of sky coordinates.
 */
static void evaluate_E_different_sky(oskar_Jones* E,
        const oskar_Sky* sky, const oskar_Telescope* telescope,
        double gast, double frequency_hz, oskar_StationWork* work,
        oskar_RandomState* rand_state, int* status)
{
    int i, num_sources, num_stations;
    const oskar_Mem *l, *m, *n;
    oskar_Mem *E_station;

    /* Check if safe to proceed. */
    if (*status) return;

    E_station = oskar_mem_create_alias(0, 0, 0, status);
    num_sources = oskar_sky_num_sources(sky);
    num_stations = oskar_telescope_num_stations(telescope);
    l = oskar_sky_l_const(sky);
    m = oskar_sky_m_const(sky);
    n = oskar_sky_n_const(sky);

    for (i = 0; i < num_stations; ++i)
    {
        const oskar_Station* station;
        station = oskar_telescope_station_const(telescope, i);
        oskar_jones_get_station_pointer(E_station, E, i, status);
        oskar_evaluate_station_beam_pattern_relative_directions(E_station,
                num_sources, l, m, n, station, work, rand_state, frequency_hz,
                gast, status);
    }
    oskar_mem_free(E_station, status);
}

#ifdef __cplusplus
}
#endif
