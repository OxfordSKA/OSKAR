/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/oskar_telescope.h"
#include "math/oskar_cmath.h"

void create_telescope_model(const char* filename, int* status)
{
    const int precision = OSKAR_DOUBLE;
    const double longitude_rad = 0.0;
    const double latitude_rad = -50.0 * M_PI / 180.0;
    const double coords_enu[][3] = {
            {0,0,0},
            {-605.1565,146.9829,0},
            {-64.79,-577.8771,0},
            {-878.9322,-123.8241,0},
            {-237.3169,-340.6209,0},
            {407.1441,198.459,0},
            {139.1693,-888.4022,0},
            {345.5998,890.5183,0},
            {-203.9332,11.27668,0},
            {-404.5717,373.44,0},
            {136.5508,745.0029,0},
            {-375.0552,-139.6862,0},
            {254.7167,-507.6426,0},
            {100.512,-355.1444,0},
            {-360.9658,161.2952,0},
            {698.7782,-471.8213,0},
            {210.4574,-100.3645,0},
            {-969.0271,190.9689,0},
            {893.7347,117.048,0},
            {-844.7219,-321.3391,0},
            {558.4537,469.6326,0},
            {-40.34307,530.6472,0},
            {-148.6963,281.2029,0},
            {-543.7159,731.1521,0},
            {256.6313,445.902,0},
            {125.0,125.0,0},
            {100.0,100.0,0},
            {50.0,50.0,0},
            {25.0,25.0,0},
            {10.0,10.0,0}
    };
    const int num_stations = sizeof(coords_enu) / sizeof(double[3]);
    oskar_Telescope* tel = oskar_telescope_create(
            precision, OSKAR_CPU, num_stations, status);
    oskar_Mem *x = 0, *y = 0, *z = 0, *error = 0;
    x = oskar_mem_create(precision, OSKAR_CPU, num_stations, status);
    y = oskar_mem_create(precision, OSKAR_CPU, num_stations, status);
    z = oskar_mem_create(precision, OSKAR_CPU, num_stations, status);
    error = oskar_mem_create(precision, OSKAR_CPU, num_stations, status);
    for (int i = 0; i < num_stations; ++i)
    {
        oskar_mem_set_element_real(x, i, coords_enu[i][0], status);
        oskar_mem_set_element_real(y, i, coords_enu[i][1], status);
        oskar_mem_set_element_real(z, i, coords_enu[i][2], status);
        oskar_mem_set_element_real(error, i, 0.0, status);
    }
    oskar_telescope_set_station_coords_enu(tel,
            longitude_rad, latitude_rad, 0.0,
            num_stations, x, y, z, error, error, error, status);
    oskar_telescope_resize_station_array(tel, 1, status);
    oskar_Station* station = oskar_telescope_station(tel, 0);
    oskar_station_resize(station, 1, status);
    oskar_telescope_save(tel, filename, status);
    oskar_telescope_free(tel, status);
    oskar_mem_free(x, status);
    oskar_mem_free(y, status);
    oskar_mem_free(z, status);
    oskar_mem_free(error, status);
}
