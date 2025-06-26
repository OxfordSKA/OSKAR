/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_save_layout(const oskar_Station* station, int feed,
        const char* filename, int* status)
{
    int i = 0;
    FILE* file = 0;
    const oskar_Mem *x_weights = 0, *y_weights = 0, *z_weights = 0;
    const oskar_Mem *x_signal = 0, *y_signal = 0, *z_signal = 0;
    if (*status || !station) return;
    const int type = oskar_station_precision(station);
    const int location = oskar_station_mem_location(station);
    const int num_elements = oskar_station_num_elements(station);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (location != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    x_weights = oskar_station_element_measured_enu_metres_const(station, feed, 0);
    y_weights = oskar_station_element_measured_enu_metres_const(station, feed, 1);
    z_weights = oskar_station_element_measured_enu_metres_const(station, feed, 2);
    x_signal = oskar_station_element_true_enu_metres_const(station, feed, 0);
    y_signal = oskar_station_element_true_enu_metres_const(station, feed, 1);
    z_signal = oskar_station_element_true_enu_metres_const(station, feed, 2);
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    (void) fprintf(file, "# Number of elements  = %i\n", num_elements);
    (void) fprintf(file, "# Longitude [radians] = %f\n", station->lon_rad);
    (void) fprintf(file, "# Latitude [radians]  = %f\n", station->lat_rad);
    (void) fprintf(file, "# Altitude [metres]   = %f\n", station->alt_metres);
    (void) fprintf(
            file, "# Local horizontal x (east), y (north), z (up) [metres], "
            "delta x, delta y, delta z [metres]\n"
    );
    if (type == OSKAR_DOUBLE)
    {
        const double *x_ = 0, *y_ = 0, *z_ = 0, *xs_ = 0, *ys_ = 0, *zs_ = 0;
        x_ = oskar_mem_double_const(x_weights, status);
        y_ = oskar_mem_double_const(y_weights, status);
        z_ = oskar_mem_double_const(z_weights, status);
        xs_ = oskar_mem_double_const(x_signal, status);
        ys_ = oskar_mem_double_const(y_signal, status);
        zs_ = oskar_mem_double_const(z_signal, status);

        for (i = 0; i < num_elements; ++i)
        {
            const double x = x_[i], y = y_[i], z = z_[i];
            (void) fprintf(
                    file, "% 14.6f % 14.6f % 14.6f % 14.6f % 14.6f % 14.6f\n",
                    x, y, z, (xs_[i] - x), (ys_[i] - y), (zs_[i] - z)
            );
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        const float *x_ = 0, *y_ = 0, *z_ = 0, *xs_ = 0, *ys_ = 0, *zs_ = 0;
        x_ = oskar_mem_float_const(x_weights, status);
        y_ = oskar_mem_float_const(y_weights, status);
        z_ = oskar_mem_float_const(z_weights, status);
        xs_ = oskar_mem_float_const(x_signal, status);
        ys_ = oskar_mem_float_const(y_signal, status);
        zs_ = oskar_mem_float_const(z_signal, status);

        for (i = 0; i < num_elements; ++i)
        {
            const float x = x_[i], y = y_[i], z = z_[i];
            (void) fprintf(
                    file, "% 14.6f % 14.6f % 14.6f % 14.6f % 14.6f % 14.6f\n",
                    x, y, z, (xs_[i] - x), (ys_[i] - y), (zs_[i] - z)
            );
        }
    }
    (void) fclose(file);
}

#ifdef __cplusplus
}
#endif
