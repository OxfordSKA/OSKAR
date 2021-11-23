/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_set_station_ids_and_coords(
        oskar_Telescope* model, int* status)
{
    if (*status) return;
    int i = 0, j = 0, counter = 0;

    /* Set coordinates into station models. */
    if (model->allow_station_beam_duplication)
    {
        /* Treat station array as holding station types only. */
        /* Set all station coordinates to that of the first station. */
        double true_geodetic[3], true_offset_ecef[3];
        for (j = 0; j < 3; ++j)
        {
            true_geodetic[j] = oskar_mem_get_element(
                    model->station_true_geodetic_rad[j], 0, status);
            true_offset_ecef[j] = oskar_mem_get_element(
                    model->station_true_offset_ecef_metres[j], 0, status);
        }
        for (i = 0; i < model->num_station_models; ++i)
        {
            oskar_station_set_position(oskar_telescope_station(model, i),
                    true_geodetic[0], true_geodetic[1], true_geodetic[2],
                    true_offset_ecef[0], true_offset_ecef[1], true_offset_ecef[2]);
        }
    }
    else
    {
        /* Populate station array for ALL stations. */
        /* Make a copy of the current station type array. */
        const int old_num_station_models = model->num_station_models;
        oskar_Station** stations = (oskar_Station**) calloc(
                old_num_station_models, sizeof(oskar_Station*));
        for (i = 0; i < old_num_station_models; ++i)
        {
            stations[i] = model->station[i];
        }

        /* Resize the station model array. */
        const int num_stations = model->num_stations;
        model->num_station_models = num_stations;
        model->station = (oskar_Station**) realloc(model->station,
                num_stations * sizeof(oskar_Station*));

        /* Copy appropriate stations from the old array using the type map. */
        const int* type_map = oskar_mem_int_const(
                oskar_telescope_station_type_map_const(model), status);
        for (i = 0; i < num_stations; ++i)
        {
            model->station[i] = oskar_station_create_copy(
                    stations[type_map[i]], model->mem_location, status);
        }

        /* Free the old station models. */
        for (i = 0; i < old_num_station_models; ++i)
        {
            oskar_station_free(stations[i], status);
        }
        free(stations);

        /* Reset the type map so we use a unique model per station. */
        oskar_telescope_set_unique_stations(model, 1, status);

        /* Set coordinates of each station. */
        double true_geodetic[3], true_offset_ecef[3];
        for (i = 0; i < model->num_stations; ++i)
        {
            for (j = 0; j < 3; ++j)
            {
                true_geodetic[j] = oskar_mem_get_element(
                        model->station_true_geodetic_rad[j], i, status);
                true_offset_ecef[j] = oskar_mem_get_element(
                        model->station_true_offset_ecef_metres[j], i, status);
            }
            oskar_station_set_position(oskar_telescope_station(model, i),
                    true_geodetic[0], true_geodetic[1], true_geodetic[2],
                    true_offset_ecef[0], true_offset_ecef[1], true_offset_ecef[2]);
        }
    }

    /* Set unique station IDs. */
    for (i = 0; i < model->num_station_models; ++i)
    {
        oskar_station_set_unique_ids(model->station[i], &counter);
    }
}

#ifdef __cplusplus
}
#endif
