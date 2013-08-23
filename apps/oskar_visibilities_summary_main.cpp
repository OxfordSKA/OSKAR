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

#include <interferometry/oskar_Visibilities.h>
#include <interferometry/oskar_visibilities_read.h>
#include <apps/lib/oskar_OptionParser.h>
#include <utility/oskar_get_error_string.h>
#include <utility/oskar_BinaryTag.h>
#include <utility/oskar_BinaryHeader.h>
#include <utility/oskar_mem_binary_stream_read.h>
#include <utility/oskar_binary_tag_index_query.h>
#include <utility/oskar_binary_tag_index_create.h>
#include <utility/oskar_binary_tag_index_free.h>
#include <utility/oskar_mem_realloc.h>

#include <utility/oskar_get_data_type_string.h>
#include <utility/oskar_mem_type_check.h>

#include <vector>
#include <string>
#include <iostream>
#include <cstdio>

using namespace std;

int main(int argc, char **argv)
{
    int status = OSKAR_SUCCESS;

    oskar_OptionParser opt("oskar_visibilities_summary");
    opt.addRequired("OSKAR visibility file");
    opt.addFlag("-l", "Display the simulation log.",
            false, "--log");
    opt.addFlag("-s", "Display the simulation settings file.",
            false, "--settings");
    if (!opt.check_options(argc, argv))
        return OSKAR_ERR_INVALID_ARGUMENT;

    bool displayLog = opt.isSet("-l") ? true : false;
    bool displaySettings = opt.isSet("-s") ? true : false;
    const char* filename = opt.getArg(0);

    oskar_Visibilities vis;
    oskar_visibilities_read(&vis, filename, &status);
    if (status)
    {
        fprintf(stderr, "ERROR: Unable to read the specified visibility file: %s\n",
                filename);
        fprintf(stderr, "ERROR: [code %i] %s.\n", status, oskar_get_error_string(status));
        return status;
    }

    if (!(displayLog || displaySettings))
    {
        printf("\n");
        printf("- Precision ............... %s\n",
                oskar_mem_is_double(vis.amplitude.type) ? "double" : "single");
        printf("- Amplitude type .......... %s\n",
                oskar_get_data_type_string(vis.amplitude.type));
        printf("\n");
        printf("- Number of stations ...... %i\n", vis.num_stations);
        printf("\n");
        printf("- Number of channels ...... %i\n", vis.num_channels);
        printf("- Number of times ......... %i\n", vis.num_times);
        printf("- Number of baselines ..... %i\n", vis.num_baselines);
        printf("- Number of polarisations . %i\n", vis.num_polarisations());
        printf("- Data order .............. %s\n",
                "{channel, time, baseline, polarisation}");
        printf("\n");
        printf("- Start frequency (MHz) ... %f\n", vis.freq_start_hz/1.e6);
        printf("- Channel separation (Hz) . %f\n", vis.freq_inc_hz);
        printf("- Channel bandwidth (Hz) .. %f\n", vis.channel_bandwidth_hz);
        printf("\n");
        printf("- Start time (MJD, UTC) ... %f\n", vis.time_start_mjd_utc);
        printf("- Time increment (s) ...... %f\n", vis.time_inc_seconds);
        printf("- Integration time (s) .... %f\n", vis.time_int_seconds);
        printf("\n");
    }

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
            oskar_Mem temp(OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
            oskar_mem_binary_stream_read(&temp, stream, &index,
                    OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0, &status);
            oskar_mem_realloc(&temp, temp.num_elements + 1, &status);
            if (status) return status;
            ((char*)temp.data)[temp.num_elements - 1] = 0;
            printf("%s", (char*)(temp.data));
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
            oskar_Mem temp(OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
            oskar_mem_binary_stream_read(&temp, stream, &index,
                    OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, 0, &status);
            oskar_mem_realloc(&temp, temp.num_elements + 1, &status);
            if (status) return status;
            ((char*)temp.data)[temp.num_elements - 1] = 0;
            printf("%s", (char*)(temp.data));
        }
        fclose(stream);
        oskar_binary_tag_index_free(index, &status);
    }

    return status;
}

