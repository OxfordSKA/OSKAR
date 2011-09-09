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

#ifndef OSKAR_TELESCOPEMODEL_H_
#define OSKAR_TELESCOPEMODEL_H_

#include "oskar_windows.h"

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_TelescopeModel_d
{
    unsigned num_antennas;
    double*  antenna_x;
    double*  antenna_y;
    double*  antenna_z; // NOTE: used after conversion to ITRS coordinates.
    double   longitude;
    double   latitude;
    bool     identical_stations; // true if all stations are identical
};
typedef struct oskar_TelescopeModel_d oskar_TelescopeModel_d;



struct oskar_TelescopeModel_f
{
    unsigned num_antennas;
    float*   antenna_x;
    float*   antenna_y;
    float*   antenna_z; // NOTE: used after conversion to ITRS coordinates.
    float    longitude;
    float    latitude;
    bool     identical_stations; // true if all stations are identical
};
typedef struct oskar_TelescopeModel_f oskar_TelescopeModel_f;


//--- Utility functions --------------------------------------------------------
DllExport
void oskar_copy_telescope_to_device_d(const oskar_TelescopeModel_d* h_telescope,
        oskar_TelescopeModel_d* hd_telescope);

DllExport
void oskar_copy_telescope_to_device_f(const oskar_TelescopeModel_f* h_telescope,
        oskar_TelescopeModel_f* hd_telescope);

DllExport
void oskar_scale_device_telescope_coords_d(oskar_TelescopeModel_d* hd_telescope,
        const double value);

DllExport
void oskar_scale_device_telescope_coords_f(oskar_TelescopeModel_f* hd_telescope,
        const float value);

DllExport
void oskar_free_device_telescope_d(oskar_TelescopeModel_d* hd_telescope);

DllExport
void oskar_free_device_telescope_f(oskar_TelescopeModel_f* hd_telescope);
//------------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif // OSKAR_TELESCOPEMODEL_H_
