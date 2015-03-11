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

#include <oskar_get_error_string.h>
#include <oskar_version_string.h>
#include <oskar_binary.h>
#include <oskar_binary_read_mem.h>
#include <oskar_log.h>

#include <apps/lib/oskar_OptionParser.h>

#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

    // Load the image into memory.
    oskar_Image* image = oskar_image_read(filename, 0, &status);

    oskar_Log* log = 0;
    if (!(displayLog || displaySettings))
    {
        oskar_log_section(log, 'M', "Image summary");
        oskar_log_value(log, 'M', 0, "File", "%s", filename);
        oskar_log_value(log, 'M', 0, "Image type", "%s", oskar_get_image_type_string(oskar_image_type(image)));
        oskar_log_value(log, 'M', 0, "Data type", "%s", oskar_mem_data_type_string(oskar_mem_type(oskar_image_data(image))));
        oskar_log_value(log, 'M', 0, "Dimension order (fastest to slowest)", 0);

        for (int i = 0; i < 5; ++i)
        {
            switch (oskar_image_dimension_order(image)[i])
            {
            case OSKAR_IMAGE_DIM_LONGITUDE:
                oskar_log_value(log, 'M', 1, 0, "[%d] Longitude", i);
                break;
            case OSKAR_IMAGE_DIM_LATITUDE:
                oskar_log_value(log, 'M', 1, 0, "[%d] Latitude", i);
                break;
            case OSKAR_IMAGE_DIM_POL:
                oskar_log_value(log, 'M', 1, 0, "[%d] Polarisation", i);
                break;
            case OSKAR_IMAGE_DIM_TIME:
                oskar_log_value(log, 'M', 1, 0, "[%d] Time", i);
                break;
            case OSKAR_IMAGE_DIM_CHANNEL:
                oskar_log_value(log, 'M', 1, 0, "[%d] Channel", i);
                break;
            default:
                /* FIXME REPLACE WITH LOG ERROR */
                fprintf(stderr, "ERROR: Unrecognised dimension ID.\n");
                return OSKAR_FAIL;
            };
        }

        oskar_log_value(log, 'M', 0, "Width, height (pixels)", "%i x %i", oskar_image_width(image), oskar_image_height(image));
        oskar_log_value(log, 'M', 0, "No. polarisations", "%i", oskar_image_num_pols(image));
        oskar_log_value(log, 'M', 0, "No. times", "%i", oskar_image_num_times(image));
        oskar_log_value(log, 'M', 0, "No. channels", "%i", oskar_image_num_channels(image));

        if (oskar_image_grid_type(image) ==
                OSKAR_IMAGE_GRID_TYPE_RECTILINEAR)
        {
            oskar_log_value(log, 'M', 0, "Field of view (deg)", "%.3f", oskar_image_fov_lat_deg(image));
            oskar_log_value(log, 'M', 0, "Centre longitude (deg)", "%.3f", oskar_image_centre_lon_deg(image));
            oskar_log_value(log, 'M', 0, "Centre latitude (deg)", "%.3f", oskar_image_centre_lat_deg(image));
        }

        oskar_log_value(log, 'M', 0, "Start time MJD(UTC)", "%.5f", oskar_image_time_start_mjd_utc(image));
        oskar_log_value(log, 'M', 0, "Time increment (seconds)", "%.1f", oskar_image_time_inc_sec(image));
        oskar_log_value(log, 'M', 0, "Start frequency (Hz)", "%e", oskar_image_freq_start_hz(image));
        oskar_log_value(log, 'M', 0, "Frequency increment (Hz)", "%e", oskar_image_freq_inc_hz(image));
    }

    // Free the image data.
    oskar_image_free(image, &status);

    // If verbose, print the run log.
    if (displayLog && !status)
    {
        oskar_Binary* h = oskar_binary_create(filename, 'r', &status);
        oskar_Mem* temp = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 1, &status);
        oskar_binary_read_mem(h, temp,
                OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0, &status);
        oskar_mem_realloc(temp, oskar_mem_length(temp) + 1, &status);
        oskar_mem_char(temp)[oskar_mem_length(temp) - 1] = 0;
        if (!status)
            printf("%s", oskar_mem_char(temp));
        status = 0;
        oskar_mem_free(temp, &status);
        oskar_binary_free(h);
    }

    if (displaySettings && !status)
    {
        oskar_Binary* h = oskar_binary_create(filename, 'r', &status);
        oskar_Mem* temp = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 1, &status);
        oskar_binary_read_mem(h, temp,
                OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, 0, &status);
        oskar_mem_realloc(temp, oskar_mem_length(temp) + 1, &status);
        oskar_mem_char(temp)[oskar_mem_length(temp) - 1] = 0;
        if (!status)
            printf("%s", oskar_mem_char(temp));
        status = 0;
        oskar_mem_free(temp, &status);
        oskar_binary_free(h);
    }

    // Check for errors.
    if (status)
    {
        fprintf(stderr, "ERROR: Run failed with code %i: %s.\n", status,
                oskar_get_error_string(status));
    }

    return status;
}
