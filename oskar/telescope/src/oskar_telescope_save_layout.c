/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_save_layout(const oskar_Telescope* telescope,
        const char* filename, int* status)
{
    int i = 0, type = 0, location = 0, num_stations = 0;
    FILE* file = 0;
    const oskar_Mem *x_weights = 0, *y_weights = 0, *z_weights = 0;
    const oskar_Mem *x_signal = 0, *y_signal = 0, *z_signal = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type and location. */
    type = oskar_telescope_precision(telescope);
    location = oskar_telescope_mem_location(telescope);
    num_stations = oskar_telescope_num_stations(telescope);
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

    /* Get pointers to the arrays. */
    x_weights = telescope->station_measured_enu_metres[0];
    y_weights = telescope->station_measured_enu_metres[1];
    z_weights = telescope->station_measured_enu_metres[2];
    x_signal = telescope->station_true_enu_metres[0];
    y_signal = telescope->station_true_enu_metres[1];
    z_signal = telescope->station_true_enu_metres[2];

    /* Open the file. */
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Save the position of each station. */
    fprintf(file, "# Number of stations  = %i\n", num_stations);
    fprintf(file, "# Longitude [radians] = %f\n", telescope->lon_rad);
    fprintf(file, "# Latitude [radians]  = %f\n", telescope->lat_rad);
    fprintf(file, "# Altitude [metres]   = %f\n", telescope->alt_metres);
    fprintf(file, "# Local horizontal x (east), y (north), z (up) [metres], "
            "delta x, delta y, delta z [metres]\n");
    if (type == OSKAR_DOUBLE)
    {
        const double *x_ = 0, *y_ = 0, *z_ = 0, *xs_ = 0, *ys_ = 0, *zs_ = 0;
        x_ = oskar_mem_double_const(x_weights, status);
        y_ = oskar_mem_double_const(y_weights, status);
        z_ = oskar_mem_double_const(z_weights, status);
        xs_ = oskar_mem_double_const(x_signal, status);
        ys_ = oskar_mem_double_const(y_signal, status);
        zs_ = oskar_mem_double_const(z_signal, status);

        for (i = 0; i < num_stations; ++i)
        {
            double x = 0.0, y = 0.0, z = 0.0;
            x = x_[i]; y = y_[i]; z = z_[i];
            fprintf(file, "% 14.6f % 14.6f % 14.6f % 14.6f % 14.6f % 14.6f\n",
                    x, y, z, (xs_[i] - x), (ys_[i] - y), (zs_[i] - z));
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

        for (i = 0; i < num_stations; ++i)
        {
            float x = 0.0, y = 0.0, z = 0.0;
            x = x_[i]; y = y_[i]; z = z_[i];
            fprintf(file, "% 14.6f % 14.6f % 14.6f % 14.6f % 14.6f % 14.6f\n",
                    x, y, z, (xs_[i] - x), (ys_[i] - y), (zs_[i] - z));
        }
    }

    /* Close the file. */
    fclose(file);
}

#ifdef __cplusplus
}
#endif
