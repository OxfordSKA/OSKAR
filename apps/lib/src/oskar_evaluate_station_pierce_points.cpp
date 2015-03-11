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


#include <oskar_evaluate_station_pierce_points.h>

#include <oskar_settings_load.h>
#include <oskar_settings_log.h>

#include <oskar_set_up_telescope.h>
#include <oskar_set_up_sky.h>

#include <oskar_log.h>
#include <oskar_sky.h>
#include <oskar_station.h>
#include <oskar_telescope.h>
#include <oskar_binary.h>
#include <oskar_binary_write_mem.h>
#include <oskar_convert_offset_ecef_to_ecef.h>
#include <oskar_convert_mjd_to_gast_fast.h>
#include <oskar_convert_apparent_ra_dec_to_enu_directions.h>
#include <oskar_evaluate_pierce_points.h>

#include <cstdio>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_station_pierce_points(const char* settings_file, oskar_Log* log)
{
    int status = 0;

    // Enum values used in writing time-freq data binary files
    enum OSKAR_TIME_FREQ_TAGS
    {
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
    oskar_settings_load(&settings, log, settings_file, &status);
    oskar_log_set_keep_file(log, settings.sim.keep_log_file);
    if (status) return status;

    oskar_Telescope* tel = oskar_set_up_telescope(&settings, log, &status);

    oskar_log_settings_telescope(log, &settings);
    oskar_log_settings_observation(log, &settings);
    oskar_log_settings_ionosphere(log, &settings);

    oskar_Sky** sky = NULL;
    int num_sky_chunks = 0;
    sky = oskar_set_up_sky(&settings, log, &num_sky_chunks, &status);

    // FIXME remove this restriction ... (loop over chunks)
    if (num_sky_chunks != 1)
        return OSKAR_ERR_SETUP_FAIL_SKY;

    // FIXME remove this restriction ... (see evaluate Z)
    if (settings.ionosphere.num_TID_screens != 1)
        return OSKAR_ERR_SETUP_FAIL;

    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int loc = OSKAR_CPU;

    oskar_Sky* chunk = sky[0];
    int num_sources = oskar_sky_num_sources(chunk);
    oskar_Mem *hor_x, *hor_y, *hor_z;
    hor_x = oskar_mem_create(type, loc, num_sources, &status);
    hor_y = oskar_mem_create(type, loc, num_sources, &status);
    hor_z = oskar_mem_create(type, loc, num_sources, &status);

    oskar_Mem *pp_lon, *pp_lat, *pp_rel_path;
    int num_stations = oskar_telescope_num_stations(tel);

    int num_pp = num_stations * num_sources;
    pp_lon = oskar_mem_create(type, loc, num_pp, &status);
    pp_lat = oskar_mem_create(type, loc, num_pp, &status);
    pp_rel_path = oskar_mem_create(type, loc, num_pp, &status);

    // Pierce points for one station (non-owned oskar_Mem pointers)
    oskar_Mem *pp_st_lon, *pp_st_lat, *pp_st_rel_path;
    pp_st_lon = oskar_mem_create_alias(0, 0, 0, &status);
    pp_st_lat = oskar_mem_create_alias(0, 0, 0, &status);
    pp_st_rel_path = oskar_mem_create_alias(0, 0, 0, &status);

    int num_times = settings.obs.num_time_steps;
    double obs_start_mjd_utc = settings.obs.start_mjd_utc;
    double dt_dump = settings.obs.dt_dump_days;

    // Binary file meta-data
    std::string label1 = "pp_lon";
    std::string label2 = "pp_lat";
    std::string label3 = "pp_path";
    std::string units  = "radians";
    std::string units2 = "";
    oskar_Mem *dims = oskar_mem_create(OSKAR_INT, loc, 2, &status);
    /* FIXME is this the correct dimension order ?
     * FIXME get the MATLAB reader to respect dimension ordering */
    oskar_mem_int(dims, &status)[0] = num_sources;
    oskar_mem_int(dims, &status)[1] = num_stations;

    const char* filename = settings.ionosphere.pierce_points.filename;
    oskar_Binary* h = oskar_binary_create(filename, 'w', &status);

    double screen_height_m = settings.ionosphere.TID->height_km * 1000.0;

//    printf("Number of times    = %i\n", num_times);
//    printf("Number of stations = %i\n", num_stations);

    void *x_, *y_, *z_;
    x_ = oskar_mem_void(oskar_telescope_station_true_x_offset_ecef_metres(tel));
    y_ = oskar_mem_void(oskar_telescope_station_true_y_offset_ecef_metres(tel));
    z_ = oskar_mem_void(oskar_telescope_station_true_z_offset_ecef_metres(tel));

    for (int t = 0; t < num_times; ++t)
    {
        double t_dump = obs_start_mjd_utc + t * dt_dump; // MJD UTC
        double gast = oskar_convert_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

        for (int i = 0; i < num_stations; ++i)
        {
            const oskar_Station* station =
                    oskar_telescope_station_const(tel, i);
            double lon = oskar_station_lon_rad(station);
            double lat = oskar_station_lat_rad(station);
            double alt = oskar_station_alt_metres(station);
            double x_ecef, y_ecef, z_ecef, x_offset, y_offset, z_offset;

            if (type == OSKAR_DOUBLE)
            {
                x_offset = ((double*)x_)[i];
                y_offset = ((double*)y_)[i];
                z_offset = ((double*)z_)[i];
            }
            else
            {
                x_offset = (double)((float*)x_)[i];
                y_offset = (double)((float*)y_)[i];
                z_offset = (double)((float*)z_)[i];
            }

            oskar_convert_offset_ecef_to_ecef(1, &x_offset, &y_offset,
                    &z_offset, lon, lat, alt, &x_ecef, &y_ecef, &z_ecef);
            double last = gast + lon;

            if (type == OSKAR_DOUBLE)
            {
                oskar_convert_apparent_ra_dec_to_enu_directions_d(num_sources,
                        oskar_mem_double_const(oskar_sky_ra_rad_const(chunk), &status),
                        oskar_mem_double_const(oskar_sky_dec_rad_const(chunk), &status),
                        last, lat, oskar_mem_double(hor_x, &status),
                        oskar_mem_double(hor_y, &status),
                        oskar_mem_double(hor_z, &status));
            }
            else
            {
                oskar_convert_apparent_ra_dec_to_enu_directions_f(num_sources,
                        oskar_mem_float_const(oskar_sky_ra_rad_const(chunk), &status),
                        oskar_mem_float_const(oskar_sky_dec_rad_const(chunk), &status),
                        last, lat, oskar_mem_float(hor_x, &status),
                        oskar_mem_float(hor_y, &status),
                        oskar_mem_float(hor_z, &status));
            }

            int offset = i * num_sources;
            oskar_mem_set_alias(pp_st_lon, pp_lon, offset, num_sources,
                    &status);
            oskar_mem_set_alias(pp_st_lat, pp_lat, offset, num_sources,
                    &status);
            oskar_mem_set_alias(pp_st_rel_path, pp_rel_path, offset,
                    num_sources, &status);
            oskar_evaluate_pierce_points(pp_st_lon, pp_st_lat, pp_st_rel_path,
                    x_ecef, y_ecef, z_ecef, screen_height_m,
                    num_sources, hor_x, hor_y, hor_z, &status);
        } // Loop over stations.

        if (status != OSKAR_SUCCESS)
            continue;

        int index = t; // could be = (num_times * f) + t if we have frequency data
        int num_fields = 3;
        int num_field_tags = 4;
        double freq_hz = 0.0;
        int freq_idx = 0;

        // Write the header TAGS
        oskar_binary_write_int(h, GRP, TIME_IDX, index, t, &status);
        oskar_binary_write_double(h, GRP, FREQ_IDX, index, freq_idx, &status);
        oskar_binary_write_double(h, GRP, TIME_MJD_UTC, index, t_dump, &status);
        oskar_binary_write_double(h, GRP, FREQ_HZ, index, freq_hz, &status);
        oskar_binary_write_int(h, GRP, NUM_FIELDS, index, num_fields, &status);
        oskar_binary_write_int(h, GRP, NUM_FIELD_TAGS, index, num_field_tags,
                &status);

        // Write data TAGS (fields)
        int field, tagID;
        field = 0;
        tagID = HEADER_OFFSET + (num_field_tags * field);
        oskar_binary_write_mem(h, pp_lon, GRP, tagID + DATA,
                index, 0, &status);
        oskar_binary_write_mem(h, dims, GRP, tagID  + DIMS,
                index, 0, &status);
        oskar_binary_write(h, OSKAR_CHAR, GRP, tagID + LABEL,
                index, label1.size()+1, label1.c_str(), &status);
        oskar_binary_write(h, OSKAR_CHAR, GRP, tagID + UNITS,
                index, units.size()+1, units.c_str(), &status);
        field = 1;
        tagID = HEADER_OFFSET + (num_field_tags * field);
        oskar_binary_write_mem(h, pp_lat, GRP, tagID + DATA,
                index, 0, &status);
        oskar_binary_write_mem(h, dims, GRP, tagID  + DIMS,
                index, 0, &status);
        oskar_binary_write(h, OSKAR_CHAR, GRP, tagID + LABEL,
                index, label2.size()+1, label2.c_str(), &status);
        oskar_binary_write(h, OSKAR_CHAR, GRP, tagID + UNITS,
                index, units.size()+1, units.c_str(), &status);
        field = 2;
        tagID = HEADER_OFFSET + (num_field_tags * field);
        oskar_binary_write_mem(h, pp_rel_path, GRP, tagID + DATA,
                index, 0, &status);
        oskar_binary_write_mem(h, dims, GRP, tagID  + DIMS,
                index, 0, &status);
        oskar_binary_write(h, OSKAR_CHAR, GRP, tagID + LABEL,
                index, label3.size()+1, label3.c_str(), &status);
        oskar_binary_write(h, OSKAR_CHAR, GRP, tagID + UNITS,
                index, units2.size()+1, units2.c_str(), &status);
    } // Loop over times

    // Close the OSKAR binary file.
    oskar_binary_free(h);

    // clean up memory
    oskar_mem_free(hor_x, &status);
    oskar_mem_free(hor_y, &status);
    oskar_mem_free(hor_z, &status);
    oskar_mem_free(pp_lon, &status);
    oskar_mem_free(pp_lat, &status);
    oskar_mem_free(pp_rel_path, &status);
    oskar_mem_free(pp_st_lon, &status);
    oskar_mem_free(pp_st_lat, &status);
    oskar_mem_free(pp_st_rel_path, &status);
    oskar_mem_free(dims, &status);
    oskar_telescope_free(tel, &status);
    oskar_sky_free(sky[0], &status);
    free(sky);

    return status;
}

#ifdef __cplusplus
}
#endif
