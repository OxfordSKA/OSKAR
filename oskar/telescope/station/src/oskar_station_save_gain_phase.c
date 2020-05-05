/*
 * Copyright (c) 2014-2020, The University of Oxford
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
    int i;
    FILE* file;
    const oskar_Mem *gain, *gain_error, *phase, *phase_error;
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
        const double *gain_, *gain_error_, *phase_, *phase_error_;
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
        const float *gain_, *gain_error_, *phase_, *phase_error_;
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
