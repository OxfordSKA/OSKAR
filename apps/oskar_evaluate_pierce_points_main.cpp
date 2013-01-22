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

#include <oskar_global.h>

#include <utility/oskar_get_error_string.h>
#include <utility/oskar_log_error.h>
#include <utility/oskar_log_message.h>
#include <utility/oskar_Log.h>
#include <utility/oskar_Settings.h>
#include <utility/oskar_Mem.h>
#include <utility/oskar_mem_init.h>
#include <utility/oskar_mem_free.h>
#include <utility/oskar_mem_get_pointer.h>

#include <interferometry/oskar_TelescopeModel.h>
#include <interferometry/oskar_telescope_model_free.h>

#include <apps/lib/oskar_settings_load.h>
#include <apps/lib/oskar_set_up_telescope.h>

#include <sky/oskar_SkyModel.h>
#include <sky/oskar_sky_model_free.h>
#include <sky/oskar_ra_dec_to_hor_lmn.h>
#include <sky/oskar_mjd_to_lmst.h>

#include <station/oskar_StationModel.h>
#include <station/oskar_evaluate_pierce_points.h>

#include <apps/lib/oskar_set_up_sky.h>

#include <cstdlib>
#include <cstdio>
#include <cmath>

static void check_error(int error, oskar_Log* log);

int main(int argc, char** argv)
{
    int error = OSKAR_SUCCESS;

    // Parse command line.
    if (argc != 2)
    {
        fprintf(stderr, "Usage: $ oskar_sim_tec_screen [settings file]\n");
        return OSKAR_ERR_INVALID_ARGUMENT;
    }

    // Create the log.
    oskar_Log log;
    oskar_log_message(&log, 0, "Running binary %s", argv[0]);
    log.keep_file = OSKAR_FALSE;

    try
    {
        // Evaluate pierce points.
        // =====================================================

        // 0) load settings.
        const char* settings_file = argv[1];
        oskar_Settings settings;
        error = oskar_settings_load(&settings, &log, settings_file);
        check_error(error, &log);
        // Force an error if not in double precision!
        if (settings.sim.double_precision != OSKAR_TRUE)
            check_error(OSKAR_ERR_BAD_DATA_TYPE, &log);

        // 1) load telescope (station positions)
        oskar_TelescopeModel tel;
        error = oskar_set_up_telescope(&tel, &log, &settings);
        check_error(error, &log);

        // 2) load sky model
        oskar_SkyModel* sky = NULL;
        int num_sky_chunks = 0;
        error = oskar_set_up_sky(&num_sky_chunks, &sky, &log, &settings);
        check_error(error, &log);
        // Force an error if more than one sky chunk.
        if (num_sky_chunks != 1)
            check_error(OSKAR_ERR_SETUP_FAIL_SKY, &log);

        double screen_height_m = 300. * 1000.;

        // 3) Compute hor_{x,y,z} - direction cosines - for sources.
        //    - this will have to be done for each station.
        oskar_SkyModel* chunk = &sky[0];
        int num_sources = chunk->num_sources;
        oskar_Mem hor_x, hor_y, hor_z;
        int type = OSKAR_DOUBLE;
        int location = OSKAR_LOCATION_CPU;
        int owner = OSKAR_TRUE;
        oskar_mem_init(&hor_x, type, location, num_sources, owner, &error);
        oskar_mem_init(&hor_y, type, location, num_sources, owner, &error);
        oskar_mem_init(&hor_z, type, location, num_sources, owner, &error);
        oskar_Mem pp_lon, pp_lat, pp_rel_path;
        int num_stations = tel.num_stations;
        int num_pp = num_stations * num_sources;
        oskar_mem_init(&pp_lon, type, location, num_pp, owner, &error);
        oskar_mem_init(&pp_lat, type, location, num_pp, owner, &error);
        oskar_mem_init(&pp_rel_path, type, location, num_pp, owner, &error);
        oskar_Mem pp_st_lon, pp_st_lat, pp_st_rel_path;
        oskar_mem_init(&pp_st_lon, type, location, num_pp, !owner, &error);
        oskar_mem_init(&pp_st_lat, type, location, num_pp, !owner, &error);
        oskar_mem_init(&pp_st_rel_path, type, location, num_pp, !owner, &error);
        double* x_ = (double*)(tel.station_x.data);
        double* y_ = (double*)(tel.station_y.data);
        double* z_ = (double*)(tel.station_z.data);
        for (int i = 0; i < num_stations; ++i)
        {
            oskar_StationModel* station = &tel.station[i];
            double lon = station->longitude_rad;
            double lat = station->latitude_rad;
            double alt = station->altitude_m;
            double mjd = 0.0;
            double x_ecef = x_[i];
            double y_ecef = y_[i];
            double z_ecef = z_[i];
            printf("r = %f\n",
                    sqrt(x_ecef*x_ecef + y_ecef*y_ecef + z_ecef*z_ecef));
            printf("station-%02i, lon = % -.4f, lat = % -.4f, "
                    "x = % -.4e, y = % -.3e, z = % -.3e (sources = %i)\n",
                    i, lon * (180./M_PI), lat * (180./M_PI),
                    x_ecef, y_ecef, z_ecef, num_sources);
            double lst = oskar_mjd_to_lmst_d(mjd, lon);
            oskar_ra_dec_to_hor_lmn_d(chunk->num_sources,
                    (double*)chunk->RA.data, (double*)chunk->Dec.data,
                    lst, lat, (double*)hor_x.data, (double*)hor_y.data,
                    (double*)hor_x.data);

            // 4) loop over telescope (stations) and compute p.p.

            int offset = i * num_sources;
            oskar_mem_get_pointer(&pp_st_lon, &pp_lon, offset, num_sources,
                    &error);
            oskar_mem_get_pointer(&pp_st_lat, &pp_lat, offset, num_sources,
                    &error);
            oskar_mem_get_pointer(&pp_st_rel_path, &pp_rel_path, offset, num_sources,
                    &error);
            oskar_evaluate_pierce_points(&pp_st_lon, &pp_st_lat, &pp_st_rel_path,
                    lon, lat, alt, x_ecef, y_ecef, z_ecef, screen_height_m,
                    num_sources, &hor_x, &hor_y, &hor_z);
        }

        // 5) save p.p. to file for plotting (lon., lat., path len.)
        double *lon_ = (double*)pp_lon.data;
        double *lat_ = (double*)pp_lat.data;
        for (int i = 0; i < num_pp; ++i)
        {
            fprintf(stderr, "%02i % 10.5f % 10.5f\n", i, lon_[i], lat_[i]);
        }

        // clean up memory
        oskar_mem_free(&hor_x, &error);
        oskar_mem_free(&hor_y, &error);
        oskar_mem_free(&hor_z, &error);
        oskar_mem_free(&pp_lon, &error);
        oskar_mem_free(&pp_lat, &error);
        oskar_mem_free(&pp_rel_path, &error);
        oskar_mem_free(&pp_st_lon, &error);
        oskar_mem_free(&pp_st_lat, &error);
        oskar_mem_free(&pp_st_rel_path, &error);
        oskar_telescope_model_free(&tel, &error);
        oskar_sky_model_free(&sky[0], &error);
        free(sky);
    }
    catch (int code)
    {
        error = code;
    }

    // Check for errors.
    check_error(error, &log);



    return error;
}


void check_error(int error, oskar_Log* log)
{
    if (error)
    {
        oskar_log_error(log, "Run failed: %s.", oskar_get_error_string(error));
        exit(error);
    }
}
