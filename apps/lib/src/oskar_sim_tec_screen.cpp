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

#include "apps/lib/oskar_set_up_telescope.h"

#include <oskar_evaluate_TEC_TID.h>
#include <oskar_SettingsIonosphere.h>
#include <oskar_mjd_to_gast_fast.h>
#include <oskar_convert_apparent_ra_dec_to_horizon_direction.h>
#include <oskar_Settings.h>
#include <oskar_mem.h>
#include <oskar_telescope.h>
#include <oskar_convert_offset_ecef_to_ecef.h>
#include <oskar_evaluate_image_lm_grid.h>
#include <oskar_image_free.h>
#include <oskar_image_init.h>
#include <oskar_image_resize.h>
#include <oskar_evaluate_image_lon_lat_grid.h>
#include <oskar_evaluate_pierce_points.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>

static void evaluate_station_beam_pp(double* pp_lon0, double* pp_lat0,
        int stationID, oskar_Settings* settings,
        oskar_Telescope* telescope, int* status);

extern "C"
int oskar_sim_tec_screen(oskar_Image* TEC_screen, oskar_Settings* settings,
        oskar_Log* log)
{
    int status = OSKAR_SUCCESS;

    oskar_SettingsIonosphere* MIM = &settings->ionosphere;

    if (!MIM->TECImage.fits_file && !MIM->TECImage.img_file)
    {
        return OSKAR_ERR_SETTINGS_IONOSPHERE;
    }

    oskar_Telescope* telescope = oskar_set_up_telescope(log, settings,
            &status);

    int im_size = MIM->TECImage.size;
    int num_pixels = im_size * im_size;

    int type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;

    int loc = OSKAR_LOCATION_CPU;

    int owner = OSKAR_TRUE;
    double fov = MIM->TECImage.fov_rad;

    // Evaluate the p.p. coordinates of the beam phase centre.
    double pp_lon0, pp_lat0;
    int st_idx = MIM->TECImage.stationID;
    if (MIM->TECImage.beam_centred)
    {
        evaluate_station_beam_pp(&pp_lon0, &pp_lat0, st_idx, settings,
                telescope, &status);
    }
    else
    {
        oskar_Station* s;
        s = oskar_telescope_station(telescope, st_idx);
        pp_lon0 = oskar_station_beam_longitude_rad(s);
        pp_lat0 = oskar_station_beam_latitude_rad(s);
    }

    int num_times = settings->obs.num_time_steps;
    double t0 = settings->obs.start_mjd_utc;
    double tinc = settings->obs.dt_dump_days;

    // Generate the lon, lat grid used for the TEC values.
    oskar_Mem pp_lon, pp_lat;
    oskar_mem_init(&pp_lon, type, loc, num_pixels, owner, &status);
    oskar_mem_init(&pp_lat, type, loc, num_pixels, owner, &status);
    oskar_evaluate_image_lon_lat_grid(&pp_lon, &pp_lat, im_size, im_size, fov,
            fov, pp_lon0, pp_lat0, &status);

    // Relative path in direction of p.p. (1.0 here as we are not using
    // any stations)
    oskar_Mem pp_rel_path;
    oskar_mem_init(&pp_rel_path, type, loc, num_pixels, owner, &status);
    oskar_mem_set_value_real(&pp_rel_path, 1.0, 0, 0, &status);

    // Initialise return values
    oskar_image_init(TEC_screen, type, loc, &status);
    oskar_image_resize(TEC_screen, im_size, im_size, 1, num_times, 1, &status);
    TEC_screen->image_type = OSKAR_IMAGE_TYPE_BEAM_SCALAR;
    TEC_screen->centre_ra_deg = pp_lon0  * (180.0/M_PI);
    TEC_screen->centre_dec_deg = pp_lat0  * (180.0/M_PI);
    TEC_screen->fov_ra_deg = fov * (180.0/M_PI);
    TEC_screen->fov_dec_deg = fov * (180.0/M_PI);
    TEC_screen->freq_start_hz = 0.0;
    TEC_screen->freq_inc_hz = 0.0;
    TEC_screen->time_inc_sec = tinc * 86400.;
    TEC_screen->time_start_mjd_utc = t0;

    oskar_Mem tec_screen;
    oskar_mem_init(&tec_screen, type, loc, num_pixels, !owner, &status);

    for (int i = 0; i < num_times; ++i)
    {
        double gast = t0 + tinc * (double)i;
        int offset = num_pixels * i;
        oskar_mem_get_pointer(&tec_screen, &(TEC_screen->data), offset,
                num_pixels, &status);
        oskar_evaluate_TEC_TID(&tec_screen, num_pixels, &pp_lon, &pp_lat,
                &pp_rel_path, MIM->TEC0, &(MIM->TID[0]), gast);
    }

    oskar_mem_free(&pp_lon, &status);
    oskar_mem_free(&pp_lat, &status);
    oskar_mem_free(&pp_rel_path, &status);
    oskar_telescope_free(telescope, &status);

    return status;
}


static void evaluate_station_beam_pp(double* pp_lon0, double* pp_lat0,
        int stationID, oskar_Settings* settings, oskar_Telescope* telescope,
        int* status)
{
    int type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;

    int loc = OSKAR_LOCATION_CPU;
    int owner = OSKAR_TRUE;

    oskar_Station* station =
            oskar_telescope_station(telescope, stationID);

    // oskar_Mem holding beam p.p. horizontal coordinates.
    oskar_Mem hor_x, hor_y, hor_z;
    oskar_mem_init(&hor_x, type, loc, 1, owner, status);
    oskar_mem_init(&hor_y, type, loc, 1, owner, status);
    oskar_mem_init(&hor_z, type, loc, 1, owner, status);

    // Offset geocentric cartesian station position
    double st_x, st_y, st_z;

    // ECEF coordinates of the station for which the beam p.p. is being evaluated.
    double st_x_ecef, st_y_ecef, st_z_ecef;

    double st_lon = oskar_station_longitude_rad(station);
    double st_lat = oskar_station_latitude_rad(station);
    double st_alt = oskar_station_altitude_m(station);

    // Time at which beam p.p. is evaluated.
    int t = 0;
    double obs_start_mjd_utc = settings->obs.start_mjd_utc;
    double dt_dump = settings->obs.dt_dump_days;
    double t_dump = obs_start_mjd_utc + t * dt_dump; // MJD UTC
    double gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);
    double last = gast + st_lon;

    void *x_, *y_, *z_;
    x_ = oskar_mem_void(oskar_telescope_station_x(telescope));
    y_ = oskar_mem_void(oskar_telescope_station_y(telescope));
    z_ = oskar_mem_void(oskar_telescope_station_z(telescope));

    if (type == OSKAR_DOUBLE)
    {
        st_x = ((double*)x_)[stationID];
        st_y = ((double*)y_)[stationID];
        st_z = ((double*)z_)[stationID];

        oskar_convert_offset_ecef_to_ecef(1, &st_x, &st_y, &st_z, st_lon,
                st_lat, st_alt, &st_x_ecef, &st_y_ecef, &st_z_ecef);

        double beam_ra = oskar_station_beam_longitude_rad(station);
        double beam_dec = oskar_station_beam_latitude_rad(station);

        // Obtain horizontal coordinates of beam p.p.
        oskar_convert_apparent_ra_dec_to_horizon_direction_d(1, &beam_ra,
                &beam_dec, last, st_lat, (double*)hor_x.data,
                (double*)hor_y.data, (double*)hor_z.data);
    }
    else // (type == OSKAR_SINGLE)
    {
        st_x = (double)((float*)x_)[stationID];
        st_y = (double)((float*)y_)[stationID];
        st_z = (double)((float*)z_)[stationID];

        oskar_convert_offset_ecef_to_ecef(1, &st_x, &st_y, &st_z, st_lon,
                st_lat, st_alt, &st_x_ecef, &st_y_ecef, &st_z_ecef);

        float beam_ra = (float)oskar_station_beam_longitude_rad(station);
        float beam_dec = (float)oskar_station_beam_latitude_rad(station);

        // Obtain horizontal coordinates of beam p.p.
        oskar_convert_apparent_ra_dec_to_horizon_direction_f(1, &beam_ra,
                &beam_dec, last, st_lat, (float*)hor_x.data,
                (float*)hor_y.data, (float*)hor_z.data);
    }

    // oskar_Mem functions holding the pp for the beam centre.
    oskar_Mem m_pp_lon0, m_pp_lat0, m_pp_rel_path;
    oskar_mem_init(&m_pp_lon0, type, loc, 1, owner, status);
    oskar_mem_init(&m_pp_lat0, type, loc, 1, owner, status);
    oskar_mem_init(&m_pp_rel_path, type, loc, 1, owner, status);

    // Pierce point of the observation phase centre - i.e. beam direction
    oskar_evaluate_pierce_points(&m_pp_lon0, &m_pp_lat0, &m_pp_rel_path,
            st_lon, st_lat, st_alt, st_x_ecef, st_y_ecef, st_z_ecef,
            settings->ionosphere.TID[0].height_km * 1000., 1, &hor_x, &hor_y,
            &hor_z, status);

    if (type == OSKAR_DOUBLE)
    {
        *pp_lon0 = ((double*)m_pp_lon0.data)[0];
        *pp_lat0 = ((double*)m_pp_lat0.data)[0];
    }
    else
    {
        *pp_lon0 = ((float*)m_pp_lon0.data)[0];
        *pp_lat0 = ((float*)m_pp_lat0.data)[0];
    }
}

