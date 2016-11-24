/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include "convert/oskar_convert_enu_to_ecef.h"
#include "convert/oskar_convert_enu_to_offset_ecef.h"
#include "convert/oskar_convert_offset_ecef_to_ecef.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_enu_to_ecef(int num_points, const double* x,
        const double* y, const double* z, double lon_rad, double lat_rad,
        double alt_metres, double* ecef_x, double* ecef_y, double* ecef_z)
{
    /* Horizon plane (ENU) to offset geocentric Cartesian coordinates. */
    oskar_convert_enu_to_offset_ecef_d(num_points, x, y, z, lon_rad, lat_rad,
            ecef_x, ecef_y, ecef_z);

    /* Offset ECEF to ECEF coordinates. */
    oskar_convert_offset_ecef_to_ecef(num_points, ecef_x, ecef_y, ecef_z,
            lon_rad, lat_rad, alt_metres, ecef_x, ecef_y, ecef_z);
}

#ifdef __cplusplus
}
#endif
