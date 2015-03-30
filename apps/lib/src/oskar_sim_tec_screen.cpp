/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include "apps/lib/oskar_set_up_telescope.h"

#include <oskar_evaluate_tec_tid.h>
#include <oskar_SettingsIonosphere.h>
#include <oskar_convert_mjd_to_gast_fast.h>
#include <oskar_convert_apparent_ra_dec_to_enu_directions.h>
#include <oskar_Settings.h>
#include <oskar_telescope.h>
#include <oskar_convert_offset_ecef_to_ecef.h>
#include <oskar_evaluate_image_lm_grid.h>
#include <oskar_image.h>
#include <oskar_evaluate_image_lon_lat_grid.h>
#include <oskar_evaluate_pierce_points.h>

#include <oskar_cmath.h>
#include <cstdlib>
#include <cstdio>

static void evaluate_station_beam_pp(double* pp_lon0, double* pp_lat0,
        int stationID, oskar_Settings* settings,
        oskar_Telescope* tel, int* status);

extern "C"
oskar_Image* oskar_sim_tec_screen(oskar_Settings* settings, oskar_Log* log,
        int* status)
{
    oskar_Image* TEC_screen = 0;
    oskar_SettingsIonosphere* MIM = &settings->ionosphere;

    if (*status) return 0;

    if (!MIM->TECImage.fits_file)
    {
        *status = OSKAR_ERR_SETTINGS_IONOSPHERE;
        return 0;
    }

    oskar_Telescope* telescope = oskar_set_up_telescope(settings, log, status);

    int im_size = MIM->TECImage.size;
    int num_pixels = im_size * im_size;
    int type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int loc = OSKAR_CPU;
    double fov = MIM->TECImage.fov_rad;

    // Evaluate the p.p. coordinates of the beam phase centre.
    double pp_lon0, pp_lat0;
    int st_idx = MIM->TECImage.stationID;
    if (MIM->TECImage.beam_centred)
    {
        evaluate_station_beam_pp(&pp_lon0, &pp_lat0, st_idx, settings,
                telescope, status);
    }
    else
    {
        oskar_Station* s = oskar_telescope_station(telescope, st_idx);
        pp_lon0 = oskar_station_beam_lon_rad(s);
        pp_lat0 = oskar_station_beam_lat_rad(s);
    }

    int num_times = settings->obs.num_time_steps;
    double t0 = settings->obs.start_mjd_utc;
    double tinc = settings->obs.dt_dump_days;

    // Generate the lon, lat grid used for the TEC values.
    oskar_Mem *pp_lon, *pp_lat, *pp_rel_path;
    pp_lon = oskar_mem_create(type, loc, num_pixels, status);
    pp_lat = oskar_mem_create(type, loc, num_pixels, status);
    oskar_evaluate_image_lon_lat_grid(pp_lon, pp_lat, im_size, im_size, fov,
            fov, pp_lon0, pp_lat0, status);

    // Relative path in direction of p.p. (1.0 here as we are not using
    // any stations)
    pp_rel_path = oskar_mem_create(type, loc, num_pixels, status);
    oskar_mem_set_value_real(pp_rel_path, 1.0, 0, 0, status);

    // Initialise return values
    TEC_screen = oskar_image_create(type, loc, status);
    oskar_image_resize(TEC_screen, im_size, im_size, 1, num_times, 1, status);
    oskar_image_set_type(TEC_screen, OSKAR_IMAGE_TYPE_BEAM_SCALAR);
    oskar_image_set_centre(TEC_screen, pp_lon0  * (180.0/M_PI),
            pp_lat0  * (180.0/M_PI));
    oskar_image_set_fov(TEC_screen, fov * (180.0/M_PI), fov * (180.0/M_PI));
    oskar_image_set_freq(TEC_screen, 0.0, 0.0);
    oskar_image_set_time(TEC_screen, t0, tinc * 86400);

    oskar_Mem *tec_screen_snapshot = oskar_mem_create_alias(0, 0, 0, status);
    for (int i = 0; i < num_times; ++i)
    {
        double gast = t0 + tinc * (double)i;
        int offset = num_pixels * i;
        oskar_mem_set_alias(tec_screen_snapshot, oskar_image_data(TEC_screen),
                offset, num_pixels, status);
        oskar_evaluate_tec_tid(tec_screen_snapshot, num_pixels, pp_lon, pp_lat,
                pp_rel_path, MIM->TEC0, &(MIM->TID[0]), gast);
    }

    oskar_mem_free(pp_lon, status);
    oskar_mem_free(pp_lat, status);
    oskar_mem_free(pp_rel_path, status);
    oskar_mem_free(tec_screen_snapshot, status);
    oskar_telescope_free(telescope, status);

    return TEC_screen;
}


static void evaluate_station_beam_pp(double* pp_lon0, double* pp_lat0,
        int stationID, oskar_Settings* settings, oskar_Telescope* tel,
        int* status)
{
    int type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int loc = OSKAR_CPU;

    oskar_Station* station = oskar_telescope_station(tel, stationID);

    // oskar_Mem holding beam p.p. horizontal coordinates.
    oskar_Mem *hor_x, *hor_y, *hor_z;
    hor_x = oskar_mem_create(type, loc, 1, status);
    hor_y = oskar_mem_create(type, loc, 1, status);
    hor_z = oskar_mem_create(type, loc, 1, status);

    // Offset geocentric cartesian station position
    double st_x, st_y, st_z;

    // ECEF coordinates of the station for which the beam p.p. is being evaluated.
    double st_x_ecef, st_y_ecef, st_z_ecef;

    double st_lon = oskar_station_lon_rad(station);
    double st_lat = oskar_station_lat_rad(station);
    double st_alt = oskar_station_alt_metres(station);

    // Time at which beam p.p. is evaluated.
    int t = 0;
    double obs_start_mjd_utc = settings->obs.start_mjd_utc;
    double dt_dump = settings->obs.dt_dump_days;
    double t_dump = obs_start_mjd_utc + t * dt_dump; // MJD UTC
    double gast = oskar_convert_mjd_to_gast_fast(t_dump + dt_dump / 2.0);
    double last = gast + st_lon;

    void *x_, *y_, *z_;
    x_ = oskar_mem_void(oskar_telescope_station_true_x_offset_ecef_metres(tel));
    y_ = oskar_mem_void(oskar_telescope_station_true_y_offset_ecef_metres(tel));
    z_ = oskar_mem_void(oskar_telescope_station_true_z_offset_ecef_metres(tel));

    if (type == OSKAR_DOUBLE)
    {
        st_x = ((double*)x_)[stationID];
        st_y = ((double*)y_)[stationID];
        st_z = ((double*)z_)[stationID];

        oskar_convert_offset_ecef_to_ecef(1, &st_x, &st_y, &st_z, st_lon,
                st_lat, st_alt, &st_x_ecef, &st_y_ecef, &st_z_ecef);

        double beam_ra = oskar_station_beam_lon_rad(station);
        double beam_dec = oskar_station_beam_lat_rad(station);

        // Obtain horizontal coordinates of beam p.p.
        oskar_convert_apparent_ra_dec_to_enu_directions_d(1, &beam_ra,
                &beam_dec, last, st_lat, oskar_mem_double(hor_x, status),
                oskar_mem_double(hor_y, status),
                oskar_mem_double(hor_z, status));
    }
    else // (type == OSKAR_SINGLE)
    {
        st_x = (double)((float*)x_)[stationID];
        st_y = (double)((float*)y_)[stationID];
        st_z = (double)((float*)z_)[stationID];

        oskar_convert_offset_ecef_to_ecef(1, &st_x, &st_y, &st_z, st_lon,
                st_lat, st_alt, &st_x_ecef, &st_y_ecef, &st_z_ecef);

        float beam_ra = (float)oskar_station_beam_lon_rad(station);
        float beam_dec = (float)oskar_station_beam_lat_rad(station);

        // Obtain horizontal coordinates of beam p.p.
        oskar_convert_apparent_ra_dec_to_enu_directions_f(1, &beam_ra,
                &beam_dec, last, st_lat, oskar_mem_float(hor_x, status),
                oskar_mem_float(hor_y, status),
                oskar_mem_float(hor_z, status));
    }

    // oskar_Mem functions holding the pp for the beam centre.
    oskar_Mem *m_pp_lon0, *m_pp_lat0, *m_pp_rel_path;
    m_pp_lon0 = oskar_mem_create(type, loc, 1, status);
    m_pp_lat0 = oskar_mem_create(type, loc, 1, status);
    m_pp_rel_path = oskar_mem_create(type, loc, 1, status);

    // Pierce point of the observation phase centre - i.e. beam direction
    oskar_evaluate_pierce_points(m_pp_lon0, m_pp_lat0, m_pp_rel_path,
            st_x_ecef, st_y_ecef, st_z_ecef,
            settings->ionosphere.TID[0].height_km * 1000., 1, hor_x, hor_y,
            hor_z, status);

    if (type == OSKAR_DOUBLE)
    {
        *pp_lon0 = oskar_mem_double(m_pp_lon0, status)[0];
        *pp_lat0 = oskar_mem_double(m_pp_lat0, status)[0];
    }
    else
    {
        *pp_lon0 = oskar_mem_float(m_pp_lon0, status)[0];
        *pp_lat0 = oskar_mem_float(m_pp_lat0, status)[0];
    }

    oskar_mem_free(m_pp_lon0, status);
    oskar_mem_free(m_pp_lat0, status);
    oskar_mem_free(m_pp_rel_path, status);
    oskar_mem_free(hor_x, status);
    oskar_mem_free(hor_y, status);
    oskar_mem_free(hor_y, status);
}

