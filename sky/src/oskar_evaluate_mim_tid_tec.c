/*
 * Copyright (c) 2013, The University of Oxford
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


#include "sky/oskar_evaluate_mim_tid_tec.h"
#include "math.h"
#include "stdio.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_tid_mim(oskar_Mem* tec, int num_directions, oskar_Mem* lon,
        oskar_Mem* lat, oskar_Mem* rel_path_length,
        oskar_SettingsMIM* settings, double gast)
{
    int i, j, type;
    double pp_lon, pp_lat;
    double pp_sec;
    double pp_tec;
    double tec0;
    double amp, w, th, v; /* TID parameters */
    double time;
    double earth_radius = 6365.0; // km -- FIXME

    /* TODO check types, dimensions etc of memory */
    type = tec->type;

    /* Loop over TIDs */
    for (i = 0; i < settings->num_tid_components; ++i)
    {
        tec0 = settings->tec0;
        amp = settings->tid[i].amp;
        // convert from km to rads
        w = settings->tid[i].wavelength / (earth_radius + settings->height_km);
        th = settings->tid[i].theta * M_PI/180.;
        // convert from km/h to rad/s
        v = (settings->tid[i].speed/(earth_radius+settings->height_km)) / 3600;

        time = gast * 86400.0; // days->sec

        /* Loop over directions */
        for (j = 0; j < num_directions; ++j)
        {
            if (type == OSKAR_DOUBLE)
            {
                pp_lon = ((double*)lon->data)[j];
                pp_lat = ((double*)lat->data)[j];
                pp_sec = ((double*)rel_path_length->data)[j];
//                printf("%f %f %f -> %f\n", 2.0*M_PI/w, cos(th) * pp_lon, v* time,
//                        (2.0*M_PI/w) * (cos(th)*pp_lon - v*time));
                pp_tec = pp_sec * amp * tec0 * (
                        cos( (2.0*M_PI/w) * (cos(th)*pp_lon - v*time) ) +
                        cos( (2.0*M_PI/w) * (sin(th)*pp_lat - v*time) )
                        );
                //pp_tec += tec0;
//                printf("c=%i d=%i %f %f %f %f\n", i, j, pp_lon, pp_lat, pp_sec, pp_tec);
                ((double*)tec->data)[j] += pp_tec;
            }
            else
            {
                pp_lon = (double)((float*)lon->data)[j];
                pp_lat = (double)((float*)lat->data)[j];
                pp_sec = (double)((float*)rel_path_length->data)[j];
                pp_tec = pp_sec * amp * tec0 * (
                        cos( (2.0*M_PI/w) * (cos(th)*pp_lon - v*time) ) +
                        cos( (2.0*M_PI/w) * (sin(th)*pp_lat - v*time) )
                );
                pp_tec += tec0;
                ((float*)tec->data)[j] += (float)pp_tec;
            }
        } // loop over directions
    } // loop over components.
}


#ifdef __cplusplus
}
#endif
