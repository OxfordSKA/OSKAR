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

#include "sky/oskar_evaluate_tec_tid.h"
#include "math/oskar_cmath.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_tec_tid(oskar_Mem* tec, int num_directions,
        const oskar_Mem* lon, const oskar_Mem* lat,
        const oskar_Mem* rel_path_length, double TEC0,
        oskar_SettingsTIDscreen* TID, double gast)
{
    int i, j, type;
    double pp_lon, pp_lat;
    double pp_sec;
    double pp_tec;
    double amp, w, th, v; /* TID parameters */
    double time;
    double earth_radius = 6365.0; /* km -- FIXME */
    int status = 0;

    /* TODO check types, dimensions etc of memory */
    type = oskar_mem_type(tec);

    oskar_mem_set_value_real(tec, 0.0, 0, oskar_mem_length(tec), &status);

    /* Loop over TIDs */
    for (i = 0; i < TID->num_components; ++i)
    {
        amp = TID->amp[i];
        /* convert from km to rads */
        w = TID->wavelength[i] / (earth_radius + TID->height_km);
        th = TID->theta[i] * M_PI/180.;
        /* convert from km/h to rad/s */
        v = (TID->speed[i]/(earth_radius + TID->height_km)) / 3600;

        time = gast * 86400.0; /* days->sec */

        /* Loop over directions */
        for (j = 0; j < num_directions; ++j)
        {
            if (type == OSKAR_DOUBLE)
            {
                pp_lon = oskar_mem_double_const(lon, &status)[j];
                pp_lat = oskar_mem_double_const(lat, &status)[j];
                pp_sec = oskar_mem_double_const(rel_path_length, &status)[j];
                pp_tec = pp_sec * amp * TEC0 * (
                        cos( (2.0*M_PI/w) * (cos(th)*pp_lon - v*time) ) +
                        cos( (2.0*M_PI/w) * (sin(th)*pp_lat - v*time) )
                        );
                pp_tec += TEC0;
                oskar_mem_double(tec, &status)[j] += pp_tec;
            }
            else
            {
                pp_lon = (double)oskar_mem_float_const(lon, &status)[j];
                pp_lat = (double)oskar_mem_float_const(lat, &status)[j];
                pp_sec = (double)oskar_mem_float_const(rel_path_length, &status)[j];
                pp_tec = pp_sec * amp * TEC0 * (
                        cos( (2.0*M_PI/w) * (cos(th)*pp_lon - v*time) ) +
                        cos( (2.0*M_PI/w) * (sin(th)*pp_lat - v*time) )
                );
                pp_tec += TEC0;
                oskar_mem_float(tec, &status)[j] += (float)pp_tec;
            }
        } /* loop over directions */
    } /* loop over components. */
}

#ifdef __cplusplus
}
#endif
