/*
 * Copyright (c) 2012-2014, The University of Oxford
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


#include <oskar_image.h>
#include <oskar_get_image_type_string.h>

#include <oskar_get_data_type_string.h>
#include <oskar_get_error_string.h>
#include <oskar_mem.h>
#include <oskar_version_string.h>

#include <oskar_BinaryTag.h>
#include <oskar_BinaryHeader.h>
#include <oskar_mem_binary_stream_read.h>
#include <oskar_binary_tag_index_query.h>
#include <oskar_binary_tag_index_free.h>
#include <oskar_binary_tag_index_query.h>
#include <oskar_binary_tag_index_create.h>
#include <oskar_binary_stream_read_oskar_version.h>
#include <oskar_log.h>

#include <apps/lib/oskar_OptionParser.h>

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const int width = 40;

int main(int argc, char** argv)
{
    int status = 0;

    oskar_OptionParser opt("oskar_image_summary", oskar_version_string());
    opt.addRequired("OSKAR image file");
    opt.addFlag("-l", "Display the simulation log associated with the image.",
            false, "--log");
    opt.addFlag("-s", "Display the image settings.", false, "--settings");
    if (!opt.check_options(argc, argv))
        return OSKAR_ERR_INVALID_ARGUMENT;

    const char* filename = opt.getArg(0);
    bool displayLog = opt.isSet("-l") ? true : false;
    bool displaySettings = opt.isSet("-s") ? true : false;

    // Get the version of OSKAR that created the file.
    int vMajor, vMinor, vPatch;
    FILE* file = fopen(filename, "rb");
    if (!file)
    {
        fprintf(stderr, "ERROR: Failed to open specified file.\n");
        return OSKAR_ERR_FILE_IO;
    }
    oskar_binary_stream_read_oskar_version(file, &vMajor, &vMinor, &vPatch,
            &status);
    // True if version 2.3 or older
    bool v232 = (vMajor <= 2 && vMinor <= 3 && vPatch <= 3) ? true : false;
    fclose(file);

    // Load the image into memory.
    oskar_Image* image = oskar_image_read(filename, 0, &status);

    oskar_Log* log = 0;
    if (!(displayLog || displaySettings))
    {
        oskar_log_section(log, "Image summary");
        oskar_log_value(log, 0, width, "File", "%s", filename);
        oskar_log_value(log, 0, width, "Created with OSKAR version",
                "%i.%i.%i", vMajor, vMinor, vPatch);
        oskar_log_value(log, 0, width, "Image type", "%s",
                oskar_get_image_type_string(oskar_image_type(image)));
        oskar_log_value(log, 0, width, "Data type", "%s",
                oskar_get_data_type_string(oskar_mem_type(
                        oskar_image_data(image))));
        oskar_log_value(log, 0, width, "Dimension order (fastest to slowest)",
                "");

        for (int i = 0; i < 5; ++i)
        {
            switch (oskar_image_dimension_order(image)[i])
            {
            case OSKAR_IMAGE_DIM_LONGITUDE:
                oskar_log_value(log, 1, width, 0, "[%d] Longitude", i);
                break;
            case OSKAR_IMAGE_DIM_LATITUDE:
                oskar_log_value(log, 1, width, 0, "[%d] Latitude", i);
                break;
            case OSKAR_IMAGE_DIM_POL:
                oskar_log_value(log, 1, width, 0, "[%d] Polarisation", i);
                break;
            case OSKAR_IMAGE_DIM_TIME:
                oskar_log_value(log, 1, width, 0, "[%d] Time", i);
                break;
            case OSKAR_IMAGE_DIM_CHANNEL:
                oskar_log_value(log, 1, width, 0, "[%d] Channel", i);
                break;
            default:
                fprintf(stderr, "ERROR: Unrecognised dimension ID.\n");
                return OSKAR_FAIL;
            };
        }

        oskar_log_value(log, 0, width, "Width, height (pixels)", "%i x %i",
                oskar_image_width(image), oskar_image_height(image));
        oskar_log_value(log, 0, width, "No. polarisations",
                "%i", oskar_image_num_pols(image));
        oskar_log_value(log, 0, width, "No. times", "%i",
                oskar_image_num_times(image));
        oskar_log_value(log, 0, width, "No. channels",
                "%i", oskar_image_num_channels(image));

        if (v232 == false)
        {
            const char* grid_type = "Undefined";
            const char* coord_frame = "Undefined";
            if (oskar_image_grid_type(image) ==
                    OSKAR_IMAGE_GRID_TYPE_RECTILINEAR)
                grid_type = "Rectilinear";
            else if (oskar_image_grid_type(image) ==
                    OSKAR_IMAGE_GRID_TYPE_HEALPIX)
                grid_type = "HEALPix";
            if (oskar_image_coord_frame(image) ==
                    OSKAR_IMAGE_COORD_FRAME_EQUATORIAL)
                coord_frame = "Equatorial (RA, Dec)";
            else if (oskar_image_coord_frame(image) ==
                    OSKAR_IMAGE_COORD_FRAME_HORIZON)
                coord_frame = "Horizon (phi, theta)";
            oskar_log_value(log, 0, width, "Grid type", "%s", grid_type);
            oskar_log_value(log, 0, width, "Coordinate frame",
                    "%s", coord_frame);
        }

        if (v232 || oskar_image_grid_type(image) ==
                OSKAR_IMAGE_GRID_TYPE_RECTILINEAR)
        {
            oskar_log_value(log, 0, width, "Field of view (deg)", "%.3f",
                    oskar_image_fov_lat_deg(image));
            oskar_log_value(log, 0, width, "Centre longitude (deg)",
                    "%.3f", oskar_image_centre_lon_deg(image));
            oskar_log_value(log, 0, width, "Centre latitude (deg)",
                    "%.3f", oskar_image_centre_lat_deg(image));
        }

        oskar_log_value(log, 0, width, "Start time MJD(UTC)",
                "%.5f", oskar_image_time_start_mjd_utc(image));
        oskar_log_value(log, 0, width, "Time increment (seconds)",
                "%.1f", oskar_image_time_inc_sec(image));
        oskar_log_value(log, 0, width, "Start frequency (Hz)",
                "%e", oskar_image_freq_start_hz(image));
        oskar_log_value(log, 0, width, "Frequency increment (Hz)",
                "%e", oskar_image_freq_inc_hz(image));
    }

    // Free the image data.
    oskar_image_free(image, &status);

    // If verbose, print the run log.
    if (displayLog)
    {
        oskar_BinaryTagIndex* index = NULL;
        FILE* stream = fopen(filename, "rb");
        if (!stream)
            return OSKAR_ERR_FILE_IO;
        oskar_binary_tag_index_create(&index, stream, &status);
        size_t data_size = 0;
        long int data_offset = 0;
        int tag_error = 0;

        oskar_binary_tag_index_query(index, OSKAR_CHAR, OSKAR_TAG_GROUP_RUN,
                OSKAR_TAG_RUN_LOG, 0, &data_size, &data_offset, &tag_error);
        if (!tag_error)
        {
            oskar_Mem *temp;
            temp = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, &status);
            oskar_mem_binary_stream_read(temp, stream, &index,
                    OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0, &status);
            oskar_mem_realloc(temp, oskar_mem_length(temp) + 1, &status);
            if (!status)
            {
                oskar_mem_char(temp)[oskar_mem_length(temp) - 1] = 0;
                printf("%s", oskar_mem_char(temp));
            }
            oskar_mem_free(temp, &status);
        }
        fclose(stream);
        oskar_binary_tag_index_free(index, &status);
    }

    if (displaySettings)
    {
        oskar_BinaryTagIndex* index = NULL;
        FILE* stream = fopen(filename, "rb");
        if (!stream)
            return OSKAR_ERR_FILE_IO;
        oskar_binary_tag_index_create(&index, stream, &status);
        size_t data_size = 0;
        long int data_offset = 0;
        int tag_error = 0;
        oskar_binary_tag_index_query(index, OSKAR_CHAR, OSKAR_TAG_GROUP_SETTINGS,
                OSKAR_TAG_SETTINGS, 0, &data_size, &data_offset, &tag_error);
        if (!tag_error)
        {
            oskar_Mem *temp;
            temp = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, &status);
            oskar_mem_binary_stream_read(temp, stream, &index,
                    OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, 0, &status);
            oskar_mem_realloc(temp, oskar_mem_length(temp) + 1, &status);
            if (!status)
            {
                oskar_mem_char(temp)[oskar_mem_length(temp) - 1] = 0;
                printf("%s", oskar_mem_char(temp));
            }
            oskar_mem_free(temp, &status);
        }
        fclose(stream);
        oskar_binary_tag_index_free(index, &status);
    }

    // Check for errors.
    if (status)
    {
        fprintf(stderr, "ERROR: Run failed with code %i: %s.\n", status,
                oskar_get_error_string(status));
    }

    return status;
}
