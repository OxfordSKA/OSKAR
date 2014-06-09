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

#include <oskar_vis.h>

#include <apps/lib/oskar_OptionParser.h>
#include <oskar_get_error_string.h>
#include <oskar_get_data_type_string.h>
#include <oskar_BinaryTag.h>
#include <oskar_BinaryHeader.h>
#include <oskar_mem_binary_stream_read.h>
#include <oskar_binary_stream_read_oskar_version.h>
#include <oskar_binary_tag_index_query.h>
#include <oskar_binary_tag_index_create.h>
#include <oskar_binary_tag_index_free.h>
#include <oskar_version_string.h>
#include <oskar_log.h>

#include <vector>
#include <string>
#include <iostream>
#include <cstdio>

static const int width = 40;

using namespace std;

int main(int argc, char **argv)
{
    int status = 0;

    oskar_OptionParser opt("oskar_vis_summary", oskar_version_string());
    opt.addRequired("OSKAR visibility file");
    opt.addFlag("-l", "Display the simulation log.",
            false, "--log");
    opt.addFlag("-s", "Display the simulation settings file.",
            false, "--settings");
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
    fclose(file);

    // Load visibilities into memory.
    oskar_Vis* vis = oskar_vis_read(filename, &status);
    if (status)
    {
        fprintf(stderr, "ERROR [code %i]: Unable to read "
                "visibility file '%s' (%s)\n", status, filename,
                oskar_get_error_string(status));
        return status;
    }

    oskar_Log* log = 0;
    if (!(displayLog || displaySettings))
    {
        oskar_log_section(log, "Visibilities summary");
        oskar_log_value(log, 0, width, "File", "%s", filename);
        oskar_log_value(log, 0, width, "Created with OSKAR version",
                "%i.%i.%i", vMajor, vMinor, vPatch);
        oskar_log_value(log, 0, width, "Precision", "%s", oskar_mem_is_double(
                oskar_vis_amplitude_const(vis)) ? "double" : "single");
        oskar_log_value(log, 0, width, "Amplitude type", "%s",
                oskar_get_data_type_string(oskar_mem_type(
                        oskar_vis_amplitude_const(vis))));
        oskar_log_value(log, 0, width, "No. stations", "%d",
                oskar_vis_num_stations(vis));
        oskar_log_value(log, 0, width, "No. channels", "%d",
                oskar_vis_num_channels(vis));
        oskar_log_value(log, 0, width, "No. times", "%d",
                oskar_vis_num_times(vis));
        oskar_log_value(log, 0, width, "No. baselines", "%d",
                oskar_vis_num_baselines(vis));
        oskar_log_value(log, 0, width, "No. polarisations", "%d",
                oskar_vis_num_polarisations(vis));
        oskar_log_value(log, 0, width, "Data order", "%s",
                "{channel, time, baseline, polarisation}");
        oskar_log_value(log, 0, width, "Start frequency (MHz)", "%.6f",
                oskar_vis_freq_start_hz(vis)/1.e6);
        oskar_log_value(log, 0, width, "Channel separation (Hz)", "%f",
                oskar_vis_freq_inc_hz(vis));
        oskar_log_value(log, 0, width, "Channel bandwidth (Hz)", "%f",
                oskar_vis_channel_bandwidth_hz(vis));
        oskar_log_value(log, 0, width, "Start time (MJD, UTC)", "%f",
                oskar_vis_time_start_mjd_utc(vis));
        oskar_log_value(log, 0, width, "Time increment (s)", "%f",
                oskar_vis_time_inc_sec(vis));
        oskar_log_value(log, 0, width, "Integration time (s)", "%f",
                oskar_vis_time_average_sec(vis));
    }
    oskar_vis_free(vis, &status);
    vis = 0;

    if (displayLog)
    {
        oskar_BinaryTagIndex* index = NULL;
        FILE* stream = fopen(filename, "rb");
        if (!stream) return OSKAR_ERR_FILE_IO;
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
        if (!stream) return OSKAR_ERR_FILE_IO;
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

    return status;
}

