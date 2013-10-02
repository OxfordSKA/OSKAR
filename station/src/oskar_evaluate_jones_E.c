/*
 * Copyright (c) 2011-2013, The University of Oxford
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
#include <oskar_evaluate_source_horizontal_lmn.h>
#include <oskar_evaluate_station_beam_aperture_array.h>
#include <oskar_evaluate_station_beam_gaussian.h>
#include <oskar_evaluate_vla_beam_pbcor.h>


#ifdef __cplusplus
extern "C" {
#endif

static void evaluate_E_common_sky_identical_stations(oskar_Jones* E,
        const oskar_Sky* sky, const oskar_Telescope* telescope,
        double gast, double frequency_hz, oskar_StationWork* work,
        oskar_RandomState* random_state, int* status);
static void evaluate_E_common_sky_different_stations(oskar_Jones* E,
        const oskar_Sky* sky, const oskar_Telescope* telescope,
        double gast, double frequency_hz, oskar_StationWork* work,
        oskar_RandomState* random_state, int* status);
static void evaluate_E_different_sky(oskar_Jones* E,
        const oskar_Sky* sky, const oskar_Telescope* telescope,
        double gast, double frequency_hz, oskar_StationWork* work,
        oskar_RandomState* random_state, int* status);

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

    /* Evaluate the station beam for each station at each source position. */
    /* A number of optimisations are possible, so switch on these. */
    if (oskar_telescope_common_horizon(telescope))
    {
        /* Optimisation 1: only evaluate one beam and copy. */
        if (oskar_telescope_identical_stations(telescope))
        {
            evaluate_E_common_sky_identical_stations(E, sky, telescope,
                    gast, frequency_hz, work, random_state, status);
        }

        /* Optimisation 2: share sky coordinates between beam evaluations */
        else /* (!telescope->identical_stations) */
        {
            evaluate_E_common_sky_different_stations(E, sky, telescope,
                    gast, frequency_hz, work, random_state, status);
        }
    }

    /* No optimisation possible.
     * Evaluate the beam per station using different sky coordinates. */
    else /* (!telescope->use_common_sky) */
    {
        evaluate_E_different_sky(E, sky, telescope, gast, frequency_hz,
                work, random_state, status);
    }
}

/*
 * Optimisation 1:
 * With a common sky (horizon) and identical stations, all station beams
 * will be the same. This function evaluates the beam once (for station 0) and
 * then copies it into the other station indices in the Jones matrix structure.
 */
static void evaluate_E_common_sky_identical_stations(oskar_Jones* E,
        const oskar_Sky* sky, const oskar_Telescope* telescope,
        double gast, double frequency_hz, oskar_StationWork* work,
        oskar_RandomState* random_state, int* status)
{
    int i, num_sources, num_stations;
    double last, lat;
    oskar_Mem E0; /* Pointer to the row of E for station 0. */
    const oskar_Station* station0;
    oskar_Mem *hor_x, *hor_y, *hor_z;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Evaluate source horizontal l,m,n once, and copy the station beam for
     * station 0 into the data for other stations in E. */
    hor_x = oskar_station_work_source_horizontal_x(work);
    hor_y = oskar_station_work_source_horizontal_y(work);
    hor_z = oskar_station_work_source_horizontal_z(work);
    station0 = oskar_telescope_station_const(telescope, 0);
    num_stations = oskar_telescope_num_stations(telescope);

    /* Evaluate the horizontal x,y,z coordinates of the sources. */
    num_sources = oskar_sky_num_sources(sky);
    last = gast + oskar_station_longitude_rad(station0);
    lat = oskar_station_latitude_rad(station0);
    oskar_evaluate_source_horizontal_lmn(num_sources, hor_x, hor_y, hor_z,
            oskar_sky_ra_const(sky), oskar_sky_dec_const(sky),
            last, lat, status);
    oskar_jones_get_station_pointer(&E0, E, 0, status);

    if (oskar_station_station_type(station0) == OSKAR_STATION_TYPE_AA)
    {
        oskar_evaluate_station_beam_aperture_array(&E0, station0, num_sources,
                hor_x, hor_y, hor_z, gast, frequency_hz, work, random_state,
                status);
    }
    else if (oskar_station_station_type(station0) ==
            OSKAR_STATION_TYPE_GAUSSIAN_BEAM)
    {
        oskar_evaluate_station_beam_gaussian(&E0, num_sources,
                oskar_sky_l_const(sky), oskar_sky_m_const(sky), hor_z,
                oskar_station_gaussian_beam_fwhm_rad(station0), status);
    }
    else if (oskar_station_station_type(station0) ==
            OSKAR_STATION_TYPE_VLA_PBCOR)
    {
        oskar_evaluate_vla_beam_pbcor(&E0, num_sources,
                oskar_sky_radius_arcmin_const(sky), frequency_hz, status);
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
    }

    /* Copy E for station 0 into memory for other stations. */
    for (i = 1; i < num_stations; ++i)
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
        const oskar_Sky* sky, const oskar_Telescope* telescope,
        double gast, double frequency_hz, oskar_StationWork* work,
        oskar_RandomState* random_state, int* status)
{
    int i, num_sources, num_stations;
    double last, lat;
    const oskar_Station* station;
    oskar_Mem *hor_x, *hor_y, *hor_z;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Evaluate horizontal x,y,z once and use them to evaluate
     * the station beam for each station. */
    hor_x = oskar_station_work_source_horizontal_x(work);
    hor_y = oskar_station_work_source_horizontal_y(work);
    hor_z = oskar_station_work_source_horizontal_z(work);
    num_stations = oskar_telescope_num_stations(telescope);
    num_sources = oskar_sky_num_sources(sky);
    station = oskar_telescope_station_const(telescope, 0);
    last = gast + oskar_station_longitude_rad(station);
    lat = oskar_station_latitude_rad(station);
    oskar_evaluate_source_horizontal_lmn(num_sources, hor_x, hor_y, hor_z,
            oskar_sky_ra_const(sky), oskar_sky_dec_const(sky),
            last, lat, status);

    for (i = 0; i < num_stations; ++i)
    {
        oskar_Mem E_station;
        station = oskar_telescope_station_const(telescope, i);
        oskar_jones_get_station_pointer(&E_station, E, i, status);

        if (oskar_station_station_type(station) == OSKAR_STATION_TYPE_AA)
        {
            oskar_evaluate_station_beam_aperture_array(&E_station, station,
                    num_sources, hor_x, hor_y, hor_z, gast, frequency_hz, work,
                    random_state, status);
        }
        else if (oskar_station_station_type(station) ==
                OSKAR_STATION_TYPE_GAUSSIAN_BEAM)
        {
            oskar_evaluate_station_beam_gaussian(&E_station, num_sources,
                    oskar_sky_l_const(sky), oskar_sky_m_const(sky), hor_z,
                    oskar_station_gaussian_beam_fwhm_rad(station), status);
        }
        else if (oskar_station_station_type(station) ==
                OSKAR_STATION_TYPE_VLA_PBCOR)
        {
            oskar_evaluate_vla_beam_pbcor(&E_station, num_sources,
                    oskar_sky_radius_arcmin_const(sky), frequency_hz, status);
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
        const oskar_Sky* sky, const oskar_Telescope* telescope,
        double gast, double frequency_hz, oskar_StationWork* work,
        oskar_RandomState* random_state, int* status)
{
    int i, num_sources, num_stations;
    oskar_Mem *hor_x, *hor_y, *hor_z;

    /* Check if safe to proceed. */
    if (*status) return;

    hor_x = oskar_station_work_source_horizontal_x(work);
    hor_y = oskar_station_work_source_horizontal_y(work);
    hor_z = oskar_station_work_source_horizontal_z(work);
    num_sources = oskar_sky_num_sources(sky);
    num_stations = oskar_telescope_num_stations(telescope);
    for (i = 0; i < num_stations; ++i)
    {
        double last, lat;
        oskar_Mem E_station;
        const oskar_Station* station;
        station = oskar_telescope_station_const(telescope, i);
        last = gast + oskar_station_longitude_rad(station);
        lat = oskar_station_latitude_rad(station);

        oskar_evaluate_source_horizontal_lmn(num_sources, hor_x, hor_y, hor_z,
                oskar_sky_ra_const(sky), oskar_sky_dec_const(sky),
                last, lat, status);
        oskar_jones_get_station_pointer(&E_station, E, i, status);

        if (oskar_station_station_type(station) == OSKAR_STATION_TYPE_AA)
        {
            oskar_evaluate_station_beam_aperture_array(&E_station, station,
                    num_sources, hor_x, hor_y, hor_z, gast, frequency_hz, work,
                    random_state, status);
        }
        else if (oskar_station_station_type(station) ==
                OSKAR_STATION_TYPE_GAUSSIAN_BEAM)
        {
            oskar_evaluate_station_beam_gaussian(&E_station, num_sources,
                    oskar_sky_l_const(sky), oskar_sky_m_const(sky), hor_z,
                    oskar_station_gaussian_beam_fwhm_rad(station), status);
        }
        else if (oskar_station_station_type(station) ==
                OSKAR_STATION_TYPE_VLA_PBCOR)
        {
            oskar_evaluate_vla_beam_pbcor(&E_station, num_sources,
                    oskar_sky_radius_arcmin_const(sky), frequency_hz, status);
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
