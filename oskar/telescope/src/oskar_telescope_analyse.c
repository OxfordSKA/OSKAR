/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#include "telescope/station/oskar_station_analyse.h"
#include "telescope/station/oskar_station_different.h"

#ifdef __cplusplus
extern "C" {
#endif

static void max_station_size_and_depth(const oskar_Station* s,
        int* max_elements, int* max_depth, int depth)
{
    if (!s) return;
    const int num_elements = oskar_station_num_elements(s);
    *max_elements = num_elements > *max_elements ? num_elements : *max_elements;
    *max_depth = depth > *max_depth ? depth : *max_depth;
    if (oskar_station_has_child(s))
    {
        int i = 0;
        for (i = 0; i < num_elements; ++i)
        {
            max_station_size_and_depth(oskar_station_child_const(s, i),
                    max_elements, max_depth, depth + 1);
        }
    }
}


static void set_child_station_locations(oskar_Station* s,
        const oskar_Station* parent)
{
    if (!s) return;
    if (parent)
    {
        oskar_station_set_position(s,
                oskar_station_lon_rad(parent),
                oskar_station_lat_rad(parent),
                oskar_station_alt_metres(parent),
                oskar_station_offset_ecef_x(parent),
                oskar_station_offset_ecef_y(parent),
                oskar_station_offset_ecef_z(parent));
    }

    /* Recursively set data for child stations. */
    if (oskar_station_has_child(s))
    {
        int i = 0, num_elements = 0;
        num_elements = oskar_station_num_elements(s);
        for (i = 0; i < num_elements; ++i)
        {
            set_child_station_locations(oskar_station_child(s, i), s);
        }
    }
}


void oskar_telescope_analyse(oskar_Telescope* model, int* status)
{
    int i = 0, finished_identical_station_check = 0, num_station_models = 0;
    if (*status) return;

    /* Recursively find the maximum number of elements in any station. */
    num_station_models = model->num_station_models;
    model->max_station_size = 0;
    for (i = 0; i < num_station_models; ++i)
    {
        set_child_station_locations(oskar_telescope_station(model, i), 0);
        max_station_size_and_depth(oskar_telescope_station_const(model, i),
                &model->max_station_size, &model->max_station_depth, 1);
    }

    /* Recursively analyse each station. */
    for (i = 0; i < num_station_models; ++i)
    {
        oskar_station_analyse(oskar_telescope_station(model, i),
                &finished_identical_station_check, status);
    }
}

#ifdef __cplusplus
}
#endif
