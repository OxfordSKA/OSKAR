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

#include "apps/lib/oskar_sim_tec_screen.h"

#include <sky/oskar_evaluate_mim_tid_tec.h>
#include <sky/oskar_SettingsMIM.h>

#include <utility/oskar_Mem.h>
#include <utility/oskar_mem_init.h>
#include <utility/oskar_mem_free.h>
#include <utility/oskar_mem_set_value_real.h>
#include <utility/oskar_mem_get_pointer.h>

#include <imaging/oskar_evaluate_image_lm_grid.h>
#include <imaging/oskar_evaluate_image_lm_grid.h>
#include <imaging/oskar_image_free.h>
#include <imaging/oskar_image_init.h>
#include <imaging/oskar_image_resize.h>

#include <fits/oskar_fits_image_write.h>

#include <math/oskar_sph_from_lm.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C"
int oskar_sim_tec_screen(const char* settings_file, oskar_Log* log)
{
    int status = OSKAR_SUCCESS;

    // Settings.
    oskar_SettingsMIM settings;
    settings.tec0 = 1.0; // 1.,5.,10.
    settings.height_km = 300.0;
    settings.num_tid_components = 3;
    settings.tid = (oskar_SettingsTID*) malloc(sizeof(oskar_SettingsTID)
            * settings.num_tid_components);
    int comp = 0;
    settings.tid[comp].amp = 0.2; // relative amplitude (to TEC0)
    settings.tid[comp].speed = 200; // km/h
    settings.tid[comp].theta = 25.0;  // degrees
    settings.tid[comp].wavelength = 300.0; // km
    comp++;
    settings.tid[comp].amp = 0.05; // relative amplitude (to TEC0)
    settings.tid[comp].speed = 300; // km/h
    settings.tid[comp].theta = -60.0;  // degrees
    settings.tid[comp].wavelength = 100.0; // km
    comp++;
    settings.tid[comp].amp = 0.3; // relative amplitude (to TEC0)
    settings.tid[comp].speed = -80; // km/h
    settings.tid[comp].theta = -40.0;  // degrees
    settings.tid[comp].wavelength = 800.0; // km
    comp++;
    settings.tid[comp].amp = 0.05; // relative amplitude (to TEC0)
    settings.tid[comp].speed = 20; // km/h
    settings.tid[comp].theta = 5.0;  // degrees
    settings.tid[comp].wavelength = 2000.0; // km

    int im_size = 512;
    int num_pixels = im_size * im_size;
    int type = OSKAR_DOUBLE;
    int loc = OSKAR_LOCATION_CPU;
    int owner = OSKAR_TRUE;
    double fov = 120. * M_PI/180.0;
    double lon0 = 0. * M_PI/180.;;
    double lat0 = 0. * M_PI/180.;

    int num_times = 400;
    double t0 = 0.0;
    double tinc = (3.0 * 60) / (86400.); // sec->days

    const char* im_file = "temp_tec_image.fits";

    // Work out the lon, lat grid used for the tec values.
    oskar_Mem grid_l, grid_m, pp_lon, pp_lat;
    oskar_mem_init(&grid_l, type, loc, num_pixels, owner, &status);
    oskar_mem_init(&grid_m, type, loc, num_pixels, owner, &status);
    oskar_mem_init(&pp_lon, type, loc, num_pixels, owner, &status);
    oskar_mem_init(&pp_lat, type, loc, num_pixels, owner, &status);
    oskar_evaluate_image_lm_grid_d(im_size, im_size, fov, fov,
            (double*)(grid_l.data), (double*)(grid_m.data));

    oskar_sph_from_lm_d(num_pixels, lon0, lat0,
            (double*)(grid_l.data), (double*)(grid_m.data),
            (double*)(pp_lon.data), (double*)(pp_lat.data));

    // Relative path in direction of pp (1.0 here as we are not using
    // any stations)
    oskar_Mem pp_rel_path;
    oskar_mem_init(&pp_rel_path, type, loc, num_pixels, owner, &status);
    oskar_mem_set_value_real(&pp_rel_path, 1.0, &status);

    // Initialise return values
    oskar_Image tec_image;
    oskar_image_init(&tec_image, type, loc, &status);
    oskar_image_resize(&tec_image, im_size, im_size, 1, num_times,
            1, &status);
    tec_image.image_type = OSKAR_IMAGE_TYPE_BEAM_SCALAR;
    tec_image.centre_ra_deg = lon0;
    tec_image.centre_dec_deg = lat0;
    tec_image.fov_ra_deg = fov * 180.0/M_PI;
    tec_image.fov_dec_deg = fov * 180.0/M_PI;
    tec_image.freq_start_hz = 0.0;
    tec_image.freq_inc_hz = 0.0;
    tec_image.time_inc_sec = tinc;
    tec_image.time_start_mjd_utc = t0;

    oskar_Mem tec_screen;
    oskar_mem_init(&tec_screen, type, loc, num_pixels, !owner, &status);

    for (int i = 0; i < num_times; ++i)
    {
        double gast = t0 + tinc * (double)i;
        int offset = num_pixels * i;
        oskar_mem_get_pointer(&tec_screen, &(tec_image.data), offset,
                num_pixels, &status);
        oskar_evaluate_tid_mim(&tec_screen, num_pixels,
                &pp_lon, &pp_lat, &pp_rel_path, &settings, gast);
    }

    if (!status)
    {
        // Write FITS image.
        status = oskar_fits_image_write(&tec_image, log, im_file);
    }

    oskar_mem_free(&grid_l, &status);
    oskar_mem_free(&grid_m, &status);
    oskar_mem_free(&pp_lon, &status);
    oskar_mem_free(&pp_lat, &status);
    oskar_mem_free(&pp_rel_path, &status);
    oskar_image_free(&tec_image, &status);
    free(settings.tid);

    return status;
}

