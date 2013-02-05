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


#include "apps/lib/oskar_evaluate_station_pierce_points.h"

#include "apps/lib/oskar_settings_load.h"
#include "apps/lib/oskar_set_up_telescope.h"
#include "apps/lib/oskar_set_up_sky.h"

#include <utility/oskar_Settings.h>
#include <utility/oskar_log_settings.h>
#include <utility/oskar_Mem.h>
#include <utility/oskar_mem_init.h>
#include <utility/oskar_mem_free.h>
#include <utility/oskar_mem_get_pointer.h>
#include <utility/oskar_mem_element_size.h>
#include <utility/oskar_binary_stream_write_header.h>
#include <utility/oskar_binary_stream_write_metadata.h>
#include <utility/oskar_binary_stream_write.h>
#include <utility/oskar_mem_binary_stream_write.h>
#include <utility/oskar_BinaryTag.h>
#include <interferometry/oskar_TelescopeModel.h>
#include <interferometry/oskar_telescope_model_free.h>
#include <interferometry/oskar_offset_geocentric_cartesian_to_geocentric_cartesian.h>
#include <sky/oskar_SkyModel.h>
#include <sky/oskar_mjd_to_gast_fast.h>
#include <sky/oskar_sky_model_free.h>
#include <sky/oskar_ra_dec_to_hor_lmn.h>
#include <station/oskar_evaluate_pierce_points.h>

#include <cstdio>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_station_pierce_points(const char* settings_file, oskar_Log* log)
{
    int status = OSKAR_SUCCESS;

    // Enum values used in writing time-freq data binary files
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

    oskar_Settings settings;
    oskar_settings_load(&settings, log, settings_file);
    log->keep_file = settings.sim.keep_log_file;

    oskar_TelescopeModel telescope;
    oskar_set_up_telescope(&telescope, log, &settings, &status);

    oskar_log_settings_telescope(log, &settings);
    oskar_log_settings_observation(log, &settings);
    oskar_log_settings_ionosphere(log, &settings);

    oskar_SkyModel* sky = NULL;
    int num_sky_chunks = 0;
    status = oskar_set_up_sky(&num_sky_chunks, &sky, log, &settings);

    // FIXME remove this restriction ... (loop over chunks)
    if (num_sky_chunks != 1)
        return OSKAR_ERR_SETUP_FAIL_SKY;

    // FIXME remove this restriction ... (see evaluate Z)
    if (settings.ionosphere.num_TID_screens != 1)
        return OSKAR_ERR_SETUP_FAIL;

    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int loc = OSKAR_LOCATION_CPU;

    oskar_SkyModel* chunk = &sky[0];
    int num_sources = chunk->num_sources;
    oskar_Mem hor_x, hor_y, hor_z;
    int owner = OSKAR_TRUE;
    oskar_mem_init(&hor_x, type, loc, num_sources, owner, &status);
    oskar_mem_init(&hor_y, type, loc, num_sources, owner, &status);
    oskar_mem_init(&hor_z, type, loc, num_sources, owner, &status);

    oskar_Mem pp_lon, pp_lat, pp_rel_path;
    int num_stations = telescope.num_stations;

    int num_pp = num_stations * num_sources;
    oskar_mem_init(&pp_lon, type, loc, num_pp, owner, &status);
    oskar_mem_init(&pp_lat, type, loc, num_pp, owner, &status);
    oskar_mem_init(&pp_rel_path, type, loc, num_pp, owner, &status);

    // Pierce points for one station (non-owned oskar_Mem pointers)
    oskar_Mem pp_st_lon, pp_st_lat, pp_st_rel_path;
    oskar_mem_init(&pp_st_lon, type, loc, num_sources, !owner, &status);
    oskar_mem_init(&pp_st_lat, type, loc, num_sources, !owner, &status);
    oskar_mem_init(&pp_st_rel_path, type, loc, num_sources, !owner, &status);

    int num_times = settings.obs.num_time_steps;
    double obs_start_mjd_utc = settings.obs.start_mjd_utc;
    double dt_dump = settings.obs.dt_dump_days;

    // Binary file meta-data
    std::string label1 = "pp_lon";
    std::string label2 = "pp_lat";
    std::string label3 = "pp_path";
    std::string units  = "radians";
    std::string units2 = "";
    oskar_Mem dims;
    oskar_mem_init(&dims, OSKAR_INT, loc, 2, owner, &status);
    /* FIXME is this the correct dimension order ?
     * FIXME get the MATLAB reader to respect dimension ordering */
    ((int*)dims.data)[0] = num_sources;
    ((int*)dims.data)[1] = num_stations;

    const char* filename = settings.ionosphere.pierce_points.filename;
    FILE* stream;
    stream = fopen(filename, "wb");
    if (stream == NULL)
        return OSKAR_ERR_FILE_IO;

    oskar_binary_stream_write_header(stream, &status);
    oskar_binary_stream_write_metadata(stream, &status);

    double screen_height_m = settings.ionosphere.TID->height_km;

//    printf("Number of times    = %i\n", num_times);
//    printf("Number of stations = %i\n", num_stations);

    for (int t = 0; t < num_times; ++t)
    {
        double t_dump = obs_start_mjd_utc + t * dt_dump; // MJD UTC
        double gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

        for (int i = 0; i < num_stations; ++i)
        {
            oskar_StationModel* station = &telescope.station[i];
            double lon = station->longitude_rad;
            double lat = station->latitude_rad;
            double alt = station->altitude_m;
            double x_ecef, y_ecef, z_ecef;
            double x_offset,y_offset,z_offset;

            if (type == OSKAR_DOUBLE)
            {
                x_offset = ((double*)telescope.station_x.data)[i];
                y_offset = ((double*)telescope.station_y.data)[i];
                z_offset = ((double*)telescope.station_z.data)[i];
            }
            else
            {
                x_offset = (double)((float*)telescope.station_x.data)[i];
                y_offset = (double)((float*)telescope.station_y.data)[i];
                z_offset = (double)((float*)telescope.station_z.data)[i];
            }

            oskar_offset_geocentric_cartesian_to_geocentric_cartesian(
                    1, &x_offset, &y_offset, &z_offset, lon, lat,
                    alt, &x_ecef, &y_ecef, &z_ecef);
            double last = gast + lon;

            if (type == OSKAR_DOUBLE)
            {
                oskar_ra_dec_to_hor_lmn_d(chunk->num_sources,
                        (double*)chunk->RA.data, (double*)chunk->Dec.data,
                        last, lat, (double*)hor_x.data, (double*)hor_y.data,
                        (double*)hor_z.data);
            }
            else
            {
                oskar_ra_dec_to_hor_lmn_f(chunk->num_sources,
                        (float*)chunk->RA.data, (float*)chunk->Dec.data,
                        last, lat, (float*)hor_x.data, (float*)hor_y.data,
                        (float*)hor_z.data);
            }

            int offset = i * num_sources;
            oskar_mem_get_pointer(&pp_st_lon, &pp_lon, offset, num_sources,
                    &status);
            oskar_mem_get_pointer(&pp_st_lat, &pp_lat, offset, num_sources,
                    &status);
            oskar_mem_get_pointer(&pp_st_rel_path, &pp_rel_path, offset, num_sources,
                    &status);
            oskar_evaluate_pierce_points(&pp_st_lon, &pp_st_lat, &pp_st_rel_path,
                    lon, lat, alt, x_ecef, y_ecef, z_ecef, screen_height_m,
                    num_sources, &hor_x, &hor_y, &hor_z, &status);
        } // Loop over stations.

        if (status != OSKAR_SUCCESS)
            continue;

        int index = t; // could be = (num_times * f) + t if we have frequency data
        int num_fields = 3;
        int num_field_tags = 4;
        double freq_hz = 0.0;
        int freq_idx = 0;

        // Write the header TAGS
        oskar_binary_stream_write_int(stream, GRP, TIME_IDX, index, t,
                &status);
        oskar_binary_stream_write_double(stream, GRP, FREQ_IDX, index,
                freq_idx, &status);
        oskar_binary_stream_write_double(stream, GRP, TIME_MJD_UTC, index,
                t_dump, &status);
        oskar_binary_stream_write_double(stream, GRP, FREQ_HZ, index,
                freq_hz, &status);
        oskar_binary_stream_write_int(stream, GRP, NUM_FIELDS, index,
                num_fields, &status);
        oskar_binary_stream_write_int(stream, GRP, NUM_FIELD_TAGS, index,
                num_field_tags, &status);

        // Write data TAGS (fields)
        int field, tagID;
        field = 0;
        tagID = HEADER_OFFSET + (num_field_tags * field);
        oskar_mem_binary_stream_write(&pp_lon, stream, GRP, tagID + DATA,
                index, 0, &status);
        oskar_mem_binary_stream_write(&dims, stream, GRP, tagID  + DIMS,
                index, 0, &status);
        oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + LABEL,
                index, label1.size()+1, label1.c_str(), &status);
        oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + UNITS,
                index, units.size()+1, units.c_str(), &status);
        field = 1;
        tagID = HEADER_OFFSET + (num_field_tags * field);
        oskar_mem_binary_stream_write(&pp_lat, stream, GRP, tagID + DATA,
                index, 0, &status);
        oskar_mem_binary_stream_write(&dims, stream, GRP, tagID  + DIMS,
                index, 0, &status);
        oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + LABEL,
                index, label2.size()+1, label2.c_str(), &status);
        oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + UNITS,
                index, units.size()+1, units.c_str(), &status);
        field = 2;
        tagID = HEADER_OFFSET + (num_field_tags * field);
        oskar_mem_binary_stream_write(&pp_rel_path, stream, GRP, tagID + DATA,
                index, 0, &status);
        oskar_mem_binary_stream_write(&dims, stream, GRP, tagID  + DIMS,
                index, 0, &status);
        oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + LABEL,
                index, label3.size()+1, label3.c_str(), &status);
        oskar_binary_stream_write(stream, OSKAR_CHAR, GRP, tagID + UNITS,
                index, units2.size()+1, units2.c_str(), &status);
    } // Loop over times

    // Close the OSKAR binary data file
    fclose(stream);

    // clean up memory
    oskar_mem_free(&hor_x, &status);
    oskar_mem_free(&hor_y, &status);
    oskar_mem_free(&hor_z, &status);
    oskar_mem_free(&pp_lon, &status);
    oskar_mem_free(&pp_lat, &status);
    oskar_mem_free(&pp_rel_path, &status);
    oskar_mem_free(&pp_st_lon, &status);
    oskar_mem_free(&pp_st_lat, &status);
    oskar_mem_free(&pp_st_rel_path, &status);
    oskar_mem_free(&dims, &status);
    //oskar_telescope_model_free(&telescope, &status);
    oskar_sky_model_free(&sky[0], &status);
    free(sky);

    return status;
}

#ifdef __cplusplus
}
#endif
