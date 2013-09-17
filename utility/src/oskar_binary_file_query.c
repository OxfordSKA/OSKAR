/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include "imaging/oskar_Image.h"
#include <oskar_vis.h>
#include <oskar_sky.h>
#include "utility/oskar_binary_stream_read.h"
#include "utility/oskar_binary_file_query.h"
#include "utility/oskar_binary_stream_read_header.h"
#include "utility/oskar_binary_header_version.h"
#include "utility/oskar_binary_tag_index_create.h"
#include "utility/oskar_binary_tag_index_free.h"
#include "utility/oskar_binary_tag_index_query.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_mem_binary_stream_read.h"
#include <oskar_log.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

static const int width = 40;

static const char* oskar_get_data_type_string(char data_type);

void oskar_binary_file_query(oskar_Log* log, const char* filename, int* status)
{
    FILE* stream;
    int version = 0, extended_tags = 0, depth = -4, i, tag_not_present = 0;
    oskar_BinaryTagIndex* index = NULL;
    oskar_BinaryHeader header;
    oskar_Mem temp;
    size_t data_size = 0;
    long int data_offset = 0;

    /* Check all inputs. */
    if (!filename || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Open the file. */
    stream = fopen(filename, "rb");
    if (!stream)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Read the header from the stream. */
    oskar_binary_stream_read_header(stream, &header, status);
    if (*status) return;
    version = oskar_binary_header_version(&header);

    /* Log file header data. */
    oskar_log_section(log, "File header in '%s'", filename);
    oskar_log_value(log, 1, width, "Binary file format version", "%d",
            header.bin_version);
    oskar_log_value(log, 1, width, "Host system is little endian", "%s",
            (header.endian == 0) ? "true" : "false");
    oskar_log_value(log, 1, width, "Size of void* on host system", "%d",
            header.size_ptr);
    oskar_log_value(log, 1, width, "Generated using OSKAR version", "%d.%d.%d",
            (version & 0xff0000) >> 16,
            (version & 0x00ff00) >> 8,
            (version & 0x0000ff));

    /* Create the tag index and close the stream. */
    oskar_binary_tag_index_create(&index, stream, status);
    oskar_log_line(log, ' ');
    oskar_log_message(log, 0, "File contains %d tags.", index->num_tags);

    /* Display the run log if it is present. */
    oskar_binary_tag_index_query(index, OSKAR_CHAR, OSKAR_TAG_GROUP_RUN,
                OSKAR_TAG_RUN_LOG, 0, &data_size, &data_offset,
                &tag_not_present);
    if (!tag_not_present)
    {
        oskar_Mem temp;
        oskar_log_section(log, "Run log:");
        oskar_mem_init(&temp, OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, 1, status);
        oskar_mem_binary_stream_read(&temp, stream, &index,
                OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0, status);
        oskar_mem_realloc(&temp, temp.num_elements + 1, status);
        ((char*)temp.data)[temp.num_elements - 1] = 0; /* Null-terminate. */
        oskar_log_message(log, depth, "\n%s", (char*)(temp.data));
        oskar_mem_free(&temp, status);
    }

    /* Iterate all tags in index. */
    oskar_log_section(log, "Standard tags:");
    oskar_log_message(log, -1, "[%3s] %-23s %5s.%-3s : %-10s (%s)",
            "ID", "TYPE", "GROUP", "TAG", "INDEX", "BYTES");
    oskar_log_message(log, depth, "CONTENTS");
    oskar_log_line(log, '-');
    for (i = 0; i < index->num_tags; ++i)
    {
        /* Check if any tags are extended. */
        if (index->extended[i])
            extended_tags++;
        else
        {
            char group, tag, type;
            int idx;
            size_t bytes;

            /* Get the tag data. */
            group = (char) (index->id_group[i]);
            tag   = (char) (index->id_tag[i]);
            type  = (char) (index->data_type[i]);
            idx   = index->user_index[i];
            bytes = index->data_size_bytes[i];
            oskar_mem_init(&temp, type, OSKAR_LOCATION_CPU, 0, 1, status);

            /* Display tag data. */
            oskar_log_message(log, -1, "[%3d] %-23s %5d.%-3d : %-10d (%ld bytes)",
                    i, oskar_get_data_type_string(type), group, tag, idx, bytes);

            /* Display more info if available. */
            if (group == OSKAR_TAG_GROUP_METADATA)
            {
                if (tag == OSKAR_TAG_METADATA_DATE_TIME_STRING)
                {
                    oskar_mem_binary_stream_read(&temp, stream, &index,
                            group, tag, idx, status);
                    oskar_log_message(log, depth, "Date: %s",
                            (char*)(temp.data));
                }
                else if (tag == OSKAR_TAG_METADATA_OSKAR_VERSION_STRING)
                {
                    oskar_mem_binary_stream_read(&temp, stream, &index,
                            group, tag, idx, status);
                    oskar_log_message(log, depth, "OSKAR version: %s",
                            (char*)(temp.data));
                }
                else if (tag == OSKAR_TAG_METADATA_CWD)
                {
                    oskar_mem_binary_stream_read(&temp, stream, &index,
                            group, tag, idx, status);
                    oskar_log_message(log, depth, "Working directory: %s",
                            (char*)(temp.data));
                }
                else if (tag == OSKAR_TAG_METADATA_USERNAME)
                {
                    oskar_mem_binary_stream_read(&temp, stream, &index,
                            group, tag, idx, status);
                    oskar_log_message(log, depth, "Username: %s",
                            (char*)(temp.data));
                }
            }
            else if (group == OSKAR_TAG_GROUP_SETTINGS)
            {
                if (tag == OSKAR_TAG_SETTINGS)
                {
                    oskar_log_message(log, depth, "Settings file");
                }
                else if (tag == OSKAR_TAG_SETTINGS_PATH)
                {
                    oskar_mem_binary_stream_read(&temp, stream, &index,
                            group, tag, idx, status);
                    oskar_log_message(log, depth, "Settings file path: %s",
                            (char*)(temp.data));
                }
            }
            else if (group == OSKAR_TAG_GROUP_RUN)
            {
                if (tag == OSKAR_TAG_RUN_LOG)
                {
                    oskar_log_message(log, depth, "Run log");
                }
            }
            else if (group == OSKAR_TAG_GROUP_IMAGE)
            {
                if (tag == OSKAR_IMAGE_TAG_IMAGE_DATA)
                {
                    oskar_log_message(log, depth, "Image data");
                }
                else if (tag == OSKAR_IMAGE_TAG_IMAGE_TYPE)
                {
                    oskar_log_message(log, depth, "Image type");
                }
                else if (tag == OSKAR_IMAGE_TAG_DATA_TYPE)
                {
                    oskar_log_message(log, depth, "Image data type");
                }
                else if (tag == OSKAR_IMAGE_TAG_DIMENSION_ORDER)
                {
                    oskar_log_message(log, depth, "Image dimension order");
                }
                else if (tag == OSKAR_IMAGE_TAG_NUM_PIXELS_WIDTH)
                {
                    int val = 0;
                    oskar_binary_stream_read_int(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "Image width: %d", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_NUM_PIXELS_HEIGHT)
                {
                    int val = 0;
                    oskar_binary_stream_read_int(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "Image height: %d", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_NUM_POLS)
                {
                    int val = 0;
                    oskar_binary_stream_read_int(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Number of polarisations: %d", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_NUM_TIMES)
                {
                    int val = 0;
                    oskar_binary_stream_read_int(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "Number of times: %d", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_NUM_CHANNELS)
                {
                    int val = 0;
                    oskar_binary_stream_read_int(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Number of channels: %d", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_CENTRE_RA)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "Centre RA [deg]: %.3f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_CENTRE_DEC)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Centre Dec [deg]: %.3f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_FOV_RA)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Field-of-view RA [deg]: %.3f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_FOV_DEC)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Field-of-view Dec [deg]: %.3f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_TIME_START_MJD_UTC)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Start time [MJD, UTC]: %.5f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_TIME_INC_SEC)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "Time inc [s]: %.1f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_FREQ_START_HZ)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Frequency start [Hz]: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_FREQ_INC_HZ)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Frequency inc [Hz]: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_MEAN)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "Mean: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_VARIANCE)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "Variance: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_MIN)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "Min: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_MAX)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "Max: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_RMS)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "RMS: %.3e", val);
                }
            }
            else if (group == OSKAR_TAG_GROUP_VISIBILITY)
            {
                if (tag == OSKAR_VIS_TAG_TELESCOPE_PATH)
                {
                    oskar_mem_binary_stream_read(&temp, stream, &index,
                            group, tag, idx, status);
                    oskar_log_message(log, depth, "Telescope model path: %s",
                            (char*)(temp.data));
                }
                else if (tag == OSKAR_VIS_TAG_NUM_CHANNELS)
                {
                    int val = 0;
                    oskar_binary_stream_read_int(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Number of channels: %d", val);
                }
                else if (tag == OSKAR_VIS_TAG_NUM_TIMES)
                {
                    int val = 0;
                    oskar_binary_stream_read_int(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "Number of times: %d", val);
                }
                else if (tag == OSKAR_VIS_TAG_NUM_BASELINES)
                {
                    int val = 0;
                    oskar_binary_stream_read_int(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Number of baselines: %d", val);
                }
                else if (tag == OSKAR_VIS_TAG_DIMENSION_ORDER)
                {
                    oskar_log_message(log, depth, "Visibility dimension order");
                }
                else if (tag == OSKAR_VIS_TAG_COORD_TYPE)
                {
                    oskar_log_message(log, depth, "Visibility coordinate type");
                }
                else if (tag == OSKAR_VIS_TAG_AMP_TYPE)
                {
                    oskar_log_message(log, depth, "Visibility amplitude type");
                }
                else if (tag == OSKAR_VIS_TAG_FREQ_START_HZ)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Frequency start [Hz]: %.3e", val);
                }
                else if (tag == OSKAR_VIS_TAG_FREQ_INC_HZ)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Frequency inc [Hz]: %.3e", val);
                }
                else if (tag == OSKAR_VIS_TAG_CHANNEL_BANDWIDTH_HZ)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Channel bandwidth [Hz]: %.3e", val);
                }
                else if (tag == OSKAR_VIS_TAG_TIME_START_MJD_UTC)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Start time [MJD, UTC]: %.5f", val);
                }
                else if (tag == OSKAR_VIS_TAG_TIME_INC_SEC)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth, "Time inc [s]: %.1f", val);
                }
                else if (tag == OSKAR_VIS_TAG_TIME_INT_SEC)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Time integration [s]: %.1f", val);
                }
                else if (tag == OSKAR_VIS_TAG_POL_TYPE)
                {
                    oskar_log_message(log, depth, "Polarisation type");
                }
                else if (tag == OSKAR_VIS_TAG_BASELINE_COORD_UNIT)
                {
                    oskar_log_message(log, depth, "Baseline coordinate unit");
                }
                else if (tag == OSKAR_VIS_TAG_BASELINE_UU)
                {
                    oskar_log_message(log, depth, "Baseline UU-coordinates");
                }
                else if (tag == OSKAR_VIS_TAG_BASELINE_VV)
                {
                    oskar_log_message(log, depth, "Baseline VV-coordinates");
                }
                else if (tag == OSKAR_VIS_TAG_BASELINE_WW)
                {
                    oskar_log_message(log, depth, "Baseline WW-coordinates");
                }
                else if (tag == OSKAR_VIS_TAG_AMPLITUDE)
                {
                    oskar_log_message(log, depth, "Visibilities");
                }
                else if (tag == OSKAR_VIS_TAG_PHASE_CENTRE_RA)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Phase centre RA [deg]: %.3f", val);
                }
                else if (tag == OSKAR_VIS_TAG_PHASE_CENTRE_DEC)
                {
                    double val = 0;
                    oskar_binary_stream_read_double(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Phase centre Dec [deg]: %.3f", val);
                }
                else if (tag == OSKAR_VIS_TAG_NUM_STATIONS)
                {
                    int val = 0;
                    oskar_binary_stream_read_int(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Number of stations: %d", val);
                }
                else if (tag == OSKAR_VIS_TAG_STATION_COORD_UNIT)
                {
                    oskar_log_message(log, depth, "Station coordinate unit");
                }
                else if (tag == OSKAR_VIS_TAG_STATION_X)
                {
                    oskar_log_message(log, depth, "Station X coordinates");
                }
                else if (tag == OSKAR_VIS_TAG_STATION_Y)
                {
                    oskar_log_message(log, depth, "Station Y coordinates");
                }
                else if (tag == OSKAR_VIS_TAG_STATION_Z)
                {
                    oskar_log_message(log, depth, "Station Z coordinates");
                }
            }
            else if (group == OSKAR_TAG_GROUP_SKY_MODEL)
            {
                if (tag == OSKAR_SKY_TAG_NUM_SOURCES)
                {
                    int val = 0;
                    oskar_binary_stream_read_int(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Number of sources: %d", val);
                }
                else if (tag == OSKAR_SKY_TAG_DATA_TYPE)
                {
                    int val = 0;
                    oskar_binary_stream_read_int(stream, &index, group, tag,
                            idx, &val, status);
                    oskar_log_message(log, depth,
                            "Data type: %s", oskar_get_data_type_string(val));
                }
                else if (tag == OSKAR_SKY_TAG_RA)
                {
                    oskar_log_message(log, depth, "Right Ascension values");
                }
                else if (tag == OSKAR_SKY_TAG_DEC)
                {
                    oskar_log_message(log, depth, "Declination values");
                }
                else if (tag == OSKAR_SKY_TAG_STOKES_I)
                {
                    oskar_log_message(log, depth, "Stokes I values");
                }
                else if (tag == OSKAR_SKY_TAG_STOKES_Q)
                {
                    oskar_log_message(log, depth, "Stokes Q values");
                }
                else if (tag == OSKAR_SKY_TAG_STOKES_U)
                {
                    oskar_log_message(log, depth, "Stokes U values");
                }
                else if (tag == OSKAR_SKY_TAG_STOKES_V)
                {
                    oskar_log_message(log, depth, "Stokes V values");
                }
                else if (tag == OSKAR_SKY_TAG_REF_FREQ)
                {
                    oskar_log_message(log, depth, "Reference frequency values");
                }
                else if (tag == OSKAR_SKY_TAG_SPECTRAL_INDEX)
                {
                    oskar_log_message(log, depth, "Spectral index values");
                }
                else if (tag == OSKAR_SKY_TAG_FWHM_MAJOR)
                {
                    oskar_log_message(log, depth, "Gaussian FWHM (major) values");
                }
                else if (tag == OSKAR_SKY_TAG_FWHM_MINOR)
                {
                    oskar_log_message(log, depth, "Gaussian FWHM (minor) values");
                }
                else if (tag == OSKAR_SKY_TAG_POSITION_ANGLE)
                {
                    oskar_log_message(log, depth, "Gaussian position angle values");
                }
            }

            /* Free temp array. */
            oskar_mem_free(&temp, status);
        }
    }

    /* Iterate extended tags in index. */
    if (extended_tags)
    {
        oskar_log_section(log, "Extended tags:");
        oskar_log_message(log, -1, "[%3s] %-23s (%s)",
                "ID", "TYPE", "BYTES");
        oskar_log_message(log, depth, "%s.%s : %s", "GROUP", "TAG", "INDEX");
        oskar_log_line(log, '-');
        for (i = 0; i < index->num_tags; ++i)
        {
            if (index->extended[i])
            {
                char *group, *tag, type;
                int idx;
                size_t bytes;

                /* Get the tag data. */
                group = index->name_group[i];
                tag   = index->name_tag[i];
                type  = (char) (index->data_type[i]);
                idx   = index->user_index[i];
                bytes = index->data_size_bytes[i];

                /* Display tag data. */
                oskar_log_message(log, -1, "[%3d] %-23s (%d bytes)",
                        i, oskar_get_data_type_string(type), bytes);
                oskar_log_message(log, depth, "%s.%s : %d", group, tag, idx);
            }
        }
    }

    /* Free the index. */
    fclose(stream);
    oskar_binary_tag_index_free(index, status);
}

static const char* oskar_get_data_type_string(char data_type)
{
    switch (data_type)
    {
        case OSKAR_CHAR:
            return "CHAR";
        case OSKAR_INT:
            return "INT";
        case OSKAR_SINGLE:
            return "SINGLE";
        case OSKAR_DOUBLE:
            return "DOUBLE";
        case OSKAR_COMPLEX:
            return "COMPLEX";
        case OSKAR_MATRIX:
            return "MATRIX";
        case OSKAR_SINGLE_COMPLEX:
            return "SINGLE COMPLEX";
        case OSKAR_DOUBLE_COMPLEX:
            return "DOUBLE COMPLEX";
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            return "SINGLE COMPLEX MATRIX";
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            return "DOUBLE COMPLEX MATRIX";
        default:
            break;
    };
    return "UNKNOWN TYPE";
}

#ifdef __cplusplus
}
#endif
