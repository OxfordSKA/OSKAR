/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "interferometer/oskar_evaluate_jones_E.h"
#include "interferometer/oskar_jones_accessors.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NOLINTNEXTLINE(readability-identifier-naming) */
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
    int i = 0, j = 0;
    if (*status) return;
    const int num_stations = oskar_telescope_num_stations(tel);
    const int num_sources = oskar_jones_num_sources(E);
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

    if (!oskar_telescope_allow_station_beam_duplication(tel))
    {
        /* Evaluate all the station beams. */
        for (i = 0; i < num_stations; ++i)
        {
            oskar_station_beam(
                    oskar_telescope_station_const(tel, i),
                    work, coord_type, num_points, source_coords,
                    ref_lon_rad, ref_lat_rad,
                    oskar_telescope_phase_centre_coord_type(tel),
                    oskar_telescope_phase_centre_longitude_rad(tel),
                    oskar_telescope_phase_centre_latitude_rad(tel),
                    time_index, gast_rad, frequency_hz,
                    i * num_sources, oskar_jones_mem(E), status);
        }
    }
    else
    {
        /* Keep track of which station models have been evaluated. */
        int num_models_evaluated = 0;
        int *models_evaluated = 0, *model_offsets = 0;
        const int* type_map = oskar_mem_int_const(
                oskar_telescope_station_type_map_const(tel), status);
        for (i = 0; i < num_stations; ++i)
        {
            int station_to_copy = -1;
            const int station_model_type = type_map[i];
            for (j = 0; j < num_models_evaluated; ++j)
            {
                if (models_evaluated[j] == station_model_type)
                {
                    station_to_copy = model_offsets[j];
                    break;
                }
            }
            if (station_to_copy >= 0)
            {
                oskar_mem_copy_contents(
                        oskar_jones_mem(E), oskar_jones_mem(E),
                        (size_t)(i * num_sources),               /* Dest. */
                        (size_t)(station_to_copy * num_sources), /* Source. */
                        (size_t)num_sources, status);
            }
            else
            {
                oskar_station_beam(
                        oskar_telescope_station_const(tel, station_model_type),
                        work, coord_type, num_points, source_coords,
                        ref_lon_rad, ref_lat_rad,
                        oskar_telescope_phase_centre_coord_type(tel),
                        oskar_telescope_phase_centre_longitude_rad(tel),
                        oskar_telescope_phase_centre_latitude_rad(tel),
                        time_index, gast_rad, frequency_hz,
                        i * num_sources, oskar_jones_mem(E), status);
                num_models_evaluated++;
                models_evaluated = (int*) realloc(models_evaluated,
                        num_models_evaluated * sizeof(int));
                model_offsets = (int*) realloc(model_offsets,
                        num_models_evaluated * sizeof(int));
                models_evaluated[num_models_evaluated - 1] = station_model_type;
                model_offsets[num_models_evaluated - 1] = i;
            }
        }
        free(models_evaluated);
        free(model_offsets);
    }
}

#ifdef __cplusplus
}
#endif
