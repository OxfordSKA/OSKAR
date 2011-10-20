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

/**
 * @file oskar_StationModel.h
 */

#include "oskar_global.h"
#include "utility/oskar_Mem.h"
#include "station/oskar_ElementModel.h"

#ifdef __cplusplus
extern "C"
#endif
struct oskar_StationModel
{
    // Station position (assumes a spherical Earth).
    double longitude;
    double latitude;
    double altitude; // Altitude above mean Earth radius of 6371.0 km.

    // Beam phase centre.
    double ra0;
    double dec0;

    // Station element position data.
    int n_elements;
    oskar_Mem x; // x-position wrt local horizon, toward the East.
    oskar_Mem y; // y-position wrt local horizon, toward the North.
    oskar_Mem z; // z-position wrt local horizon, toward the zenith.
    oskar_Mem weight;
    oskar_Mem amp_gain;
    oskar_Mem amp_error;
    oskar_Mem phase_offset;
    oskar_Mem phase_error;
    oskar_StationModel* station; // NULL when there are no child stations.

    // Embedded element pattern.
    int n_element_patterns; // Only for when there are no child stations.
    oskar_ElementModel* element_pattern; // NULL if there are child stations.

    int bit_depth; // Not implemented!
};

typedef struct oskar_StationModel oskar_StationModel;



// DEPRECATED

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
void oskar_station_model_copy_to_device_d(const oskar_StationModel_d* h_stations,
        const unsigned num_stations, oskar_StationModel_d* hd_stations);

OSKAR_EXPORT
void oskar_station_model_copy_to_device_f(const oskar_StationModel_f* h_stations,
        const unsigned num_stations, oskar_StationModel_f* hd_stations);

OSKAR_EXPORT
void oskar_station_model_scale_coords_d(const unsigned num_stations,
        oskar_StationModel_d* hd_stations, const double value);

OSKAR_EXPORT
void oskar_station_model_scale_coords_f(const unsigned num_stations,
        oskar_StationModel_f* hd_stations, const float value);
//------------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif // OSKAR_STATION_MODEL_H_
