/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/oskar_station.h"
#include "math/oskar_cmath.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define R2D (180.0 / M_PI)

void oskar_station_save_gain_phase(const oskar_Station* station, int feed,
        const char* filename, int* status)
{
    int i = 0;
    FILE* file = 0;
    const oskar_Mem *gain = 0, *gain_error = 0, *phase = 0, *phase_error = 0;
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
    gain = oskar_station_element_gain_const(station, feed);
    gain_error = oskar_station_element_gain_error_const(station, feed);
    phase = oskar_station_element_phase_offset_rad_const(station, feed);
    phase_error = oskar_station_element_phase_error_rad_const(station, feed);
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    if (type == OSKAR_DOUBLE)
    {
        const double *gain_ = 0, *gain_error_ = 0;
        const double *phase_ = 0, *phase_error_ = 0;
        gain_ = oskar_mem_double_const(gain, status);
        gain_error_ = oskar_mem_double_const(gain_error, status);
        phase_ = oskar_mem_double_const(phase, status);
        phase_error_ = oskar_mem_double_const(phase_error, status);

        for (i = 0; i < num_elements; ++i)
        {
            fprintf(file, "% 14.6f % 14.6f % 14.6f % 14.6f\n",
                    gain_[i], phase_[i] * R2D,
                    gain_error_[i], phase_error_[i] * R2D);
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        const float *gain_ = 0, *gain_error_ = 0;
        const float *phase_ = 0, *phase_error_ = 0;
        gain_ = oskar_mem_float_const(gain, status);
        gain_error_ = oskar_mem_float_const(gain_error, status);
        phase_ = oskar_mem_float_const(phase, status);
        phase_error_ = oskar_mem_float_const(phase_error, status);

        for (i = 0; i < num_elements; ++i)
        {
            fprintf(file, "% 14.6f % 14.6f % 14.6f % 14.6f\n",
                    gain_[i], phase_[i] * R2D,
                    gain_error_[i], phase_error_[i] * R2D);
        }
    }
    fclose(file);
}

#ifdef __cplusplus
}
#endif
