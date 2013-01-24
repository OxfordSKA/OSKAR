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

#include <utility/oskar_BinaryTag.h>
#include "utility/oskar_mem_binary_stream_write.h"
#include <utility/oskar_binary_stream_write.h>
#include <utility/oskar_binary_stream_write_header.h>
#include <utility/oskar_binary_stream_write_metadata.h>

#include <interferometry/oskar_TelescopeModel.h>
#include <interferometry/oskar_telescope_model_free.h>
#include <interferometry/oskar_offset_geocentric_cartesian_to_geocentric_cartesian.h>

#include <apps/lib/oskar_settings_load.h>
#include <apps/lib/oskar_set_up_telescope.h>

#include <sky/oskar_SkyModel.h>
#include <sky/oskar_sky_model_free.h>
#include <sky/oskar_ra_dec_to_hor_lmn.h>
#include <sky/oskar_mjd_to_last_fast.h>
#include <sky/oskar_mjd_to_gast_fast.h>

#include <station/oskar_StationModel.h>
#include <station/oskar_evaluate_pierce_points.h>

#include <apps/lib/oskar_set_up_sky.h>

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>

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
        // TODO load ionospheric (Z-Jones) settings.
        // Need to define an output for pierce points.
        const char* settings_file = argv[1];
        oskar_Settings settings;
        error = oskar_settings_load(&settings, &log, settings_file);
        check_error(error, &log);
        // Force an error if not in double precision!
        if (settings.sim.double_precision != OSKAR_TRUE)
            check_error(OSKAR_ERR_BAD_DATA_TYPE, &log);

        log.keep_file = OSKAR_FALSE;

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

        // TODO handle time properly and add time loop.
        int num_times = settings.obs.num_time_steps;
        double obs_start_mjd_utc = settings.obs.start_mjd_utc;
        double dt_dump = settings.obs.dt_dump_days;

//        printf("start time = %f days\n", obs_start_mjd_utc);
//        printf("t_inc = %f days\n", dt_dump);

        // Open file to write (stream) to.
        const char* filename = "temp_pp.dat";
        FILE* stream;
        stream = fopen(filename, "wb");
        if (stream == NULL)
            check_error(OSKAR_ERR_FILE_IO, &log);
        oskar_binary_stream_write_header(stream, &error);
        oskar_binary_stream_write_metadata(stream, &error);
        enum {
            TIME_IDX       = 0,
            FREQ_IDX       = 1,
            TIME_MJD_UTC   = 2,
            FREQ_HZ        = 3,
            NUM_FIELDS     = 4,
            NUM_FIELD_TAGS = 5,
            HEADER_OFFSET  = 10,
            DATA           = 0,
            DIMS           = 1,
            LABEL          = 2,
            UNITS          = 3,
            GRP            = OSKAR_TAG_GROUP_TIME_FREQ_DATA
        };

        std::string label1 = "pp_lon";
        std::string label2 = "pp_lat";
        std::string label3 = "pp_path";
        std::string units = "radians";
        std::string units2 = "";
        oskar_Mem dims1, dims2;
        oskar_mem_init(&dims1, OSKAR_INT, location, 2, owner, &error);
        oskar_mem_init(&dims2, OSKAR_INT, location, 2, owner, &error);
        ((int*)dims1.data)[0] = num_pp;
        ((int*)dims1.data)[1] = 1;
        ((int*)dims2.data)[0] = num_pp;
        ((int*)dims2.data)[1] = 1;

        for (int t = 0; t < num_times; ++t)
        {
            double t_dump = obs_start_mjd_utc + t * dt_dump; // MJD UTC
//            printf(">>>>>>>> time = %f MJD <<<<<<<<\n", t_dump+dt_dump/2.0);
            double gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);
//            printf(">>>>>>>> time = %f gast <<<<<<<<\n", gast);
//            printf("\n");

            // TODO function to convert gast to date time (or mjd to date time)
            // (so it can be displayed.)

            //printf("time %i - %f\n", t, gast);

            for (int i = 0; i < num_stations; ++i)
            {
                oskar_StationModel* station = &tel.station[i];
                double lon = station->longitude_rad;
                double lat = station->latitude_rad;
                double alt = station->altitude_m;
                double x_ecef, y_ecef, z_ecef;
                oskar_offset_geocentric_cartesian_to_geocentric_cartesian(
                        1, &x_[i], &y_[i], &z_[i], lon, lat,
                        alt, &x_ecef, &y_ecef, &z_ecef);
                double last = gast + lon;
                oskar_ra_dec_to_hor_lmn_d(chunk->num_sources,
                        (double*)chunk->RA.data, (double*)chunk->Dec.data,
                        last, lat, (double*)hor_x.data, (double*)hor_y.data,
                        (double*)hor_z.data);

//                printf("> st %i\n", i);
//                printf("lon=%f, gast=%f, last=%f, lat=%f\n",
//                        lon*180./M_PI, gast,
//                        last, lat*180./M_PI);
//                printf("ra=%f, dec=%f, x=%f, y=%f, z=%f\n",
//                        ((double*)chunk->RA.data)[0]*180./M_PI,
//                        ((double*)chunk->Dec.data)[0]*180./M_PI,
//                        ((double*)hor_x.data)[0],
//                        ((double*)hor_y.data)[0],
//                        ((double*)hor_z.data)[0]
//                );

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
                printf("\n");
            }

            if (error) continue;

            int index = t; // could be = (num_times * f) + t if we have frequency data
            const int num_fields = 3;
            const int num_field_tags = 4;
            double freq_hz = 0.0;
            int freq_idx = 0;

            // Write the header TAGS
            oskar_binary_stream_write_int(stream, GRP, TIME_IDX, index, t,
                    &error);
            oskar_binary_stream_write_double(stream, GRP, FREQ_IDX, index,
                    freq_idx, &error);
            oskar_binary_stream_write_double(stream, GRP, TIME_MJD_UTC, index,
                    t_dump, &error);
            oskar_binary_stream_write_double(stream, GRP, FREQ_HZ, index,
                    freq_hz, &error);
            oskar_binary_stream_write_int(stream, GRP, NUM_FIELDS, index,
                    num_fields, &error);
            oskar_binary_stream_write_int(stream, GRP, NUM_FIELD_TAGS, index,
                    num_field_tags, &error);

            // Write data TAGS (fields)
            int field, tagID;
            field = 0;
            tagID = HEADER_OFFSET + (num_field_tags * field);
            oskar_mem_binary_stream_write(&pp_lon, stream, GRP, tagID + DATA,
                    index, 0, &error);
            oskar_mem_binary_stream_write(&dims1, stream, GRP, tagID  + DIMS,
                    index, 0, &error);
            oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + LABEL,
                    index, label1.size()+1, label1.c_str(), &error);
            oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + UNITS,
                    index, units.size()+1, units.c_str(), &error);
            field = 1;
            tagID = HEADER_OFFSET + (num_field_tags * field);
            oskar_mem_binary_stream_write(&pp_lat, stream, GRP, tagID + DATA,
                    index, 0, &error);
            oskar_mem_binary_stream_write(&dims2, stream, GRP, tagID  + DIMS,
                    index, 0, &error);
            oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + LABEL,
                    index, label2.size()+1, label2.c_str(), &error);
            oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + UNITS,
                    index, units.size()+1, units.c_str(), &error);
            field = 2;
            tagID = HEADER_OFFSET + (num_field_tags * field);
            oskar_mem_binary_stream_write(&pp_rel_path, stream, GRP, tagID + DATA,
                    index, 0, &error);
            oskar_mem_binary_stream_write(&dims2, stream, GRP, tagID  + DIMS,
                    index, 0, &error);
            oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + LABEL,
                    index, label3.size()+1, label3.c_str(), &error);
            oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + UNITS,
                    index, units2.size()+1, units2.c_str(), &error);

            if (error)
                printf("Something when wrong writing binary data...\n");

        }

        // Close the OSKAR binary data file
        fclose(stream);

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
        oskar_mem_free(&dims1, &error);
        oskar_mem_free(&dims2, &error);
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
