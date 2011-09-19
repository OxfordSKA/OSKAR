/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_STATION_MODEL_H_
#define OSKAR_STATION_MODEL_H_

#include "oskar_global.h"


#ifdef __cplusplus
extern "C" {
#endif

struct oskar_StationModel_d
{
    unsigned num_antennas;
    double*  antenna_x;
    double*  antenna_y;
    //double * antenna_z;
};
typedef struct oskar_StationModel_d oskar_StationModel_d;

struct oskar_StationModel_f
{
    unsigned num_antennas;
    float*   antenna_x;
    float*   antenna_y;
    //double * antenna_z;
};
typedef struct oskar_StationModel_f oskar_StationModel_f;

//----- Utility functions -----------------------------------------------------
OSKAR_EXPORT
void oskar_copy_stations_to_device_d(const oskar_StationModel_d* h_stations,
        const unsigned num_stations, oskar_StationModel_d* hd_stations);

OSKAR_EXPORT
void oskar_copy_stations_to_device_f(const oskar_StationModel_f* h_stations,
        const unsigned num_stations, oskar_StationModel_f* hd_stations);

OSKAR_EXPORT
void oskar_scale_station_coords_d(const unsigned num_stations,
        oskar_StationModel_d* hd_stations, const double value);

OSKAR_EXPORT
void oskar_scale_station_coords_f(const unsigned num_stations,
        oskar_StationModel_f* hd_stations, const float value);
//------------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif // OSKAR_STATION_MODEL_H_
