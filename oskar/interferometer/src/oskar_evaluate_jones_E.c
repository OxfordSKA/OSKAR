/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "interferometer/oskar_evaluate_jones_E.h"
#include "interferometer/oskar_jones_accessors.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_jones_E(
        oskar_Jones* E,
        int coord_type,
        int num_points,
        const oskar_Mem* const source_coords[3],
        double ref_lon_rad,
        double ref_lat_rad,
        const oskar_Telescope* tel,
        int time_index,
        double gast_rad,
        double frequency_hz,
        oskar_StationWork* work,
        int* status)
{
    int i;
    if (*status) return;
    const int num_stations = oskar_telescope_num_stations(tel);
    const int num_sources = oskar_jones_num_sources(E);
    const int n = (oskar_telescope_allow_station_beam_duplication(tel) &&
            oskar_telescope_identical_stations(tel)) ? 1 : num_stations;
    if (num_stations == 0)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }
    if (num_stations != oskar_jones_num_stations(E))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Evaluate the station beam(s). */
    for (i = 0; i < n; ++i)
        oskar_station_beam(
                oskar_telescope_station_const(tel, i), work,
                coord_type, num_points, source_coords,
                ref_lon_rad, ref_lat_rad,
                oskar_telescope_phase_centre_coord_type(tel),
                oskar_telescope_phase_centre_longitude_rad(tel),
                oskar_telescope_phase_centre_latitude_rad(tel),
                time_index, gast_rad, frequency_hz,
                i * num_sources, oskar_jones_mem(E), status);

    /* Copy station beam only if required. */
    for (i = n; i < num_stations; ++i)
        oskar_mem_copy_contents(
                oskar_jones_mem(E), oskar_jones_mem(E),
                (size_t)(i * num_sources), 0,
                (size_t)num_sources, status);
}

#ifdef __cplusplus
}
#endif
