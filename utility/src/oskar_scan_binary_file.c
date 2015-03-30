/*
 * Copyright (c) 2012-2015, The University of Oxford
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

/* FIXME Remove? */
#include <private_binary.h>

#include <oskar_image.h>
#include <oskar_vis.h>
#include <oskar_sky.h>
#include <oskar_log.h>
#include <oskar_binary.h>
#include <oskar_binary_read_mem.h>

#include <oskar_scan_binary_file.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_scan_binary_file(oskar_Log* log, const char* filename, int* status)
{
    int extended_tags = 0, depth = -4, i, tag_not_present = 0;
    oskar_Binary* h = NULL;
    oskar_Mem* temp = 0;
    char p = 'M'; /* Log entry priority */

    /* Check if safe to proceed. */
    if (*status) return;

    /* Open the file. */
    h = oskar_binary_create(filename, 'r', status);
    if (*status)
    {
        oskar_binary_free(h);
        return;
    }

    /* Log file header data. */
    oskar_log_section(log, p, "File header in '%s'", filename);
    oskar_log_value(log, p, 1, "Binary file format version", "%d",
            h->bin_version);
    oskar_log_line(log, p, ' ');
    oskar_log_message(log, p, 0, "File contains %d chunks.", h->num_chunks);

    /* Display the run log if it is present. */
    temp = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    oskar_binary_read_mem(h, temp, OSKAR_TAG_GROUP_RUN,
            OSKAR_TAG_RUN_LOG, 0, &tag_not_present);
    oskar_mem_realloc(temp, oskar_mem_length(temp) + 1, status);
    oskar_mem_char(temp)[oskar_mem_length(temp) - 1] = 0; /* Null-terminate. */
    if (!tag_not_present)
    {
        oskar_log_section(log, p, "Run log:");
        oskar_log_message(log, p, -1, "\n%s", oskar_mem_char(temp));
    }
    oskar_mem_free(temp, status);

    /* Iterate all tags in index. */
    oskar_log_section(log, p, "Standard tags:");
    oskar_log_message(log, p, -1, "[%3s] %-23s %5s.%-3s : %-10s (%s)",
            "ID", "TYPE", "GROUP", "TAG", "INDEX", "BYTES");
    oskar_log_message(log, p, depth, "CONTENTS");
    oskar_log_line(log, p, '-');
    for (i = 0; i < h->num_chunks; ++i)
    {
        /* Check if any tags are extended. */
        if (h->extended[i])
            extended_tags++;
        else
        {
            char group, tag, type;
            int idx;
            size_t bytes;

            /* Get the tag data. */
            group = (char) (h->id_group[i]);
            tag   = (char) (h->id_tag[i]);
            type  = (char) (h->data_type[i]);
            idx   = h->user_index[i];
            bytes = h->payload_size_bytes[i];
            temp = oskar_mem_create(type, OSKAR_CPU, 0, status);

            /* Display tag data. */
            oskar_log_message(log, p, -1, "[%3d] %-23s %5d.%-3d : %-10d (%ld bytes)",
                    i, oskar_mem_data_type_string(type), group, tag, idx, bytes);

            /* Display more info if available. */
            if (group == OSKAR_TAG_GROUP_METADATA)
            {
                if (tag == OSKAR_TAG_METADATA_DATE_TIME_STRING)
                {
                    oskar_binary_read_mem(h, temp, group, tag, idx, status);
                    oskar_log_message(log, p, depth, "Date: %s",
                            oskar_mem_char(temp));
                }
                else if (tag == OSKAR_TAG_METADATA_OSKAR_VERSION_STRING)
                {
                    oskar_binary_read_mem(h, temp, group, tag, idx, status);
                    oskar_log_message(log, p, depth, "OSKAR version: %s",
                            oskar_mem_char(temp));
                }
                else if (tag == OSKAR_TAG_METADATA_CWD)
                {
                    oskar_binary_read_mem(h, temp, group, tag, idx, status);
                    oskar_log_message(log, p, depth, "Working directory: %s",
                            oskar_mem_char(temp));
                }
                else if (tag == OSKAR_TAG_METADATA_USERNAME)
                {
                    oskar_binary_read_mem(h, temp, group, tag, idx, status);
                    oskar_log_message(log, p, depth, "Username: %s",
                            oskar_mem_char(temp));
                }
            }
            else if (group == OSKAR_TAG_GROUP_SETTINGS)
            {
                if (tag == OSKAR_TAG_SETTINGS)
                {
                    oskar_log_message(log, p, depth, "Settings file");
                }
                else if (tag == OSKAR_TAG_SETTINGS_PATH)
                {
                    oskar_binary_read_mem(h, temp, group, tag, idx, status);
                    oskar_log_message(log, p, depth, "Settings file path: %s",
                            oskar_mem_char(temp));
                }
            }
            else if (group == OSKAR_TAG_GROUP_RUN)
            {
                if (tag == OSKAR_TAG_RUN_LOG)
                {
                    oskar_log_message(log, p, depth, "Run log");
                }
            }
            else if (group == OSKAR_TAG_GROUP_IMAGE)
            {
                if (tag == OSKAR_IMAGE_TAG_IMAGE_DATA)
                {
                    oskar_log_message(log, p, depth, "Image data");
                }
                else if (tag == OSKAR_IMAGE_TAG_IMAGE_TYPE)
                {
                    oskar_log_message(log, p, depth, "Image type");
                }
                else if (tag == OSKAR_IMAGE_TAG_DATA_TYPE)
                {
                    oskar_log_message(log, p, depth, "Image data type");
                }
                else if (tag == OSKAR_IMAGE_TAG_DIMENSION_ORDER)
                {
                    oskar_log_message(log, p, depth, "Image dimension order");
                }
                else if (tag == OSKAR_IMAGE_TAG_NUM_PIXELS_WIDTH)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Image width: %d", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_NUM_PIXELS_HEIGHT)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Image height: %d", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_NUM_POLS)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Number of polarisations: %d", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_NUM_TIMES)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Number of times: %d", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_NUM_CHANNELS)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Number of channels: %d", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_CENTRE_LONGITUDE)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Centre longitude [deg]: %.3f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_CENTRE_LATITUDE)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Centre latitude [deg]: %.3f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_FOV_LONGITUDE)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Field-of-view longitude [deg]: %.3f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_FOV_LATITUDE)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Field-of-view latitude [deg]: %.3f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_TIME_START_MJD_UTC)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Start time [MJD, UTC]: %.5f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_TIME_INC_SEC)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Time inc [s]: %.1f", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_FREQ_START_HZ)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Frequency start [Hz]: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_FREQ_INC_HZ)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Frequency inc [Hz]: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_MEAN)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Mean: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_VARIANCE)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Variance: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_MIN)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Min: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_MAX)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Max: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_RMS)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "RMS: %.3e", val);
                }
                else if (tag == OSKAR_IMAGE_TAG_GRID_TYPE)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    if (val == OSKAR_IMAGE_GRID_TYPE_RECTILINEAR)
                        oskar_log_message(log, p, depth, "Grid type: Rectilinear");
                    else if (val == OSKAR_IMAGE_GRID_TYPE_HEALPIX)
                        oskar_log_message(log, p, depth, "Grid type: HEALPix Ring");
                    else
                        oskar_log_message(log, p, depth, "Grid type: Undef");
                }
                else if (tag == OSKAR_IMAGE_TAG_COORD_FRAME)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    if (val == OSKAR_IMAGE_COORD_FRAME_EQUATORIAL)
                        oskar_log_message(log, p, depth, "Coordinate frame: Equatorial");
                    else if (val == OSKAR_IMAGE_COORD_FRAME_HORIZON)
                        oskar_log_message(log, p, depth, "Coordinate frame: Horizon");
                    else
                        oskar_log_message(log, p, depth, "Coordinate frame: Undef");
                }
                else if (tag == OSKAR_IMAGE_TAG_HEALPIX_NSIDE)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "HEALPix nside: %i", val);
                }
            }
            else if (group == OSKAR_TAG_GROUP_VISIBILITY)
            {
                if (tag == OSKAR_VIS_TAG_TELESCOPE_PATH)
                {
                    oskar_binary_read_mem(h, temp, group, tag, idx, status);
                    oskar_log_message(log, p, depth, "Telescope model path: %s",
                            oskar_mem_char(temp));
                }
                else if (tag == OSKAR_VIS_TAG_NUM_CHANNELS)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Number of channels: %d", val);
                }
                else if (tag == OSKAR_VIS_TAG_NUM_TIMES)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Number of times: %d", val);
                }
                else if (tag == OSKAR_VIS_TAG_NUM_BASELINES)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Number of baselines: %d", val);
                }
                else if (tag == OSKAR_VIS_TAG_DIMENSION_ORDER)
                {
                    oskar_log_message(log, p, depth, "Visibility dimension order");
                }
                else if (tag == OSKAR_VIS_TAG_COORD_TYPE)
                {
                    oskar_log_message(log, p, depth, "Visibility coordinate type");
                }
                else if (tag == OSKAR_VIS_TAG_AMP_TYPE)
                {
                    oskar_log_message(log, p, depth, "Visibility amplitude type");
                }
                else if (tag == OSKAR_VIS_TAG_FREQ_START_HZ)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Frequency start [Hz]: %.3e", val);
                }
                else if (tag == OSKAR_VIS_TAG_FREQ_INC_HZ)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Frequency inc [Hz]: %.3e", val);
                }
                else if (tag == OSKAR_VIS_TAG_CHANNEL_BANDWIDTH_HZ)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Channel bandwidth [Hz]: %.3e", val);
                }
                else if (tag == OSKAR_VIS_TAG_TIME_START_MJD_UTC)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Start time [MJD, UTC]: %.5f", val);
                }
                else if (tag == OSKAR_VIS_TAG_TIME_INC_SEC)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth, "Time inc [s]: %.1f", val);
                }
                else if (tag == OSKAR_VIS_TAG_TIME_AVERAGE_SEC)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Time average integration [s]: %.1f", val);
                }
                else if (tag == OSKAR_VIS_TAG_POL_TYPE)
                {
                    oskar_log_message(log, p, depth, "Polarisation type");
                }
                else if (tag == OSKAR_VIS_TAG_BASELINE_COORD_UNIT)
                {
                    oskar_log_message(log, p, depth, "Baseline coordinate unit");
                }
                else if (tag == OSKAR_VIS_TAG_BASELINE_UU)
                {
                    oskar_log_message(log, p, depth, "Baseline UU-coordinates");
                }
                else if (tag == OSKAR_VIS_TAG_BASELINE_VV)
                {
                    oskar_log_message(log, p, depth, "Baseline VV-coordinates");
                }
                else if (tag == OSKAR_VIS_TAG_BASELINE_WW)
                {
                    oskar_log_message(log, p, depth, "Baseline WW-coordinates");
                }
                else if (tag == OSKAR_VIS_TAG_AMPLITUDE)
                {
                    oskar_log_message(log, p, depth, "Visibilities");
                }
                else if (tag == OSKAR_VIS_TAG_PHASE_CENTRE_RA)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Phase centre RA [deg]: %.3f", val);
                }
                else if (tag == OSKAR_VIS_TAG_PHASE_CENTRE_DEC)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Phase centre Dec [deg]: %.3f", val);
                }
                else if (tag == OSKAR_VIS_TAG_TELESCOPE_LON)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Telescope longitude [deg]: %.3f", val);
                }
                else if (tag == OSKAR_VIS_TAG_TELESCOPE_LAT)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Telescope latitude [deg]: %.3f", val);
                }
                else if (tag == OSKAR_VIS_TAG_TELESCOPE_ALT)
                {
                    double val = 0;
                    oskar_binary_read_double(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Telescope altitude [m]: %.3f", val);
                }
                else if (tag == OSKAR_VIS_TAG_NUM_STATIONS)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Number of stations: %d", val);
                }
                else if (tag == OSKAR_VIS_TAG_STATION_COORD_UNIT)
                {
                    oskar_log_message(log, p, depth, "Station coordinate unit");
                }
                else if (tag == OSKAR_VIS_TAG_STATION_X_OFFSET_ECEF)
                {
                    oskar_log_message(log, p, depth,
                            "Station X coordinates (offset ECEF)");
                }
                else if (tag == OSKAR_VIS_TAG_STATION_Y_OFFSET_ECEF)
                {
                    oskar_log_message(log, p, depth,
                            "Station Y coordinates (offset ECEF)");
                }
                else if (tag == OSKAR_VIS_TAG_STATION_Z_OFFSET_ECEF)
                {
                    oskar_log_message(log, p, depth,
                            "Station Z coordinates (offset ECEF)");
                }
                else if (tag == OSKAR_VIS_TAG_STATION_X_ENU)
                {
                    oskar_log_message(log, p, depth,
                            "Station X coordinates (ENU)");
                }
                else if (tag == OSKAR_VIS_TAG_STATION_Y_ENU)
                {
                    oskar_log_message(log, p, depth,
                            "Station Y coordinates (ENU)");
                }
                else if (tag == OSKAR_VIS_TAG_STATION_Z_ENU)
                {
                    oskar_log_message(log, p, depth,
                            "Station Z coordinates (ENU)");
                }
            }
            else if (group == OSKAR_TAG_GROUP_SKY_MODEL)
            {
                if (tag == OSKAR_SKY_TAG_NUM_SOURCES)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Number of sources: %d", val);
                }
                else if (tag == OSKAR_SKY_TAG_DATA_TYPE)
                {
                    int val = 0;
                    oskar_binary_read_int(h, group, tag, idx, &val, status);
                    oskar_log_message(log, p, depth,
                            "Data type: %s", oskar_mem_data_type_string(val));
                }
                else if (tag == OSKAR_SKY_TAG_RA)
                {
                    oskar_log_message(log, p, depth, "Right Ascension values");
                }
                else if (tag == OSKAR_SKY_TAG_DEC)
                {
                    oskar_log_message(log, p, depth, "Declination values");
                }
                else if (tag == OSKAR_SKY_TAG_STOKES_I)
                {
                    oskar_log_message(log, p, depth, "Stokes I values");
                }
                else if (tag == OSKAR_SKY_TAG_STOKES_Q)
                {
                    oskar_log_message(log, p, depth, "Stokes Q values");
                }
                else if (tag == OSKAR_SKY_TAG_STOKES_U)
                {
                    oskar_log_message(log, p, depth, "Stokes U values");
                }
                else if (tag == OSKAR_SKY_TAG_STOKES_V)
                {
                    oskar_log_message(log, p, depth, "Stokes V values");
                }
                else if (tag == OSKAR_SKY_TAG_REF_FREQ)
                {
                    oskar_log_message(log, p, depth, "Reference frequency values");
                }
                else if (tag == OSKAR_SKY_TAG_SPECTRAL_INDEX)
                {
                    oskar_log_message(log, p, depth, "Spectral index values");
                }
                else if (tag == OSKAR_SKY_TAG_ROTATION_MEASURE)
                {
                    oskar_log_message(log, p, depth, "Rotation measure values");
                }
                else if (tag == OSKAR_SKY_TAG_FWHM_MAJOR)
                {
                    oskar_log_message(log, p, depth, "Gaussian FWHM (major) values");
                }
                else if (tag == OSKAR_SKY_TAG_FWHM_MINOR)
                {
                    oskar_log_message(log, p, depth, "Gaussian FWHM (minor) values");
                }
                else if (tag == OSKAR_SKY_TAG_POSITION_ANGLE)
                {
                    oskar_log_message(log, p, depth, "Gaussian position angle values");
                }
            }

            /* Free temp array. */
            oskar_mem_free(temp, status);
        }
    }

    /* Iterate extended tags in index. */
    if (extended_tags)
    {
        oskar_log_section(log, p, "Extended tags:");
        oskar_log_message(log, p, -1, "[%3s] %-23s (%s)",
                "ID", "TYPE", "BYTES");
        oskar_log_message(log, p, depth, "%s.%s : %s", "GROUP", "TAG", "INDEX");
        oskar_log_line(log, p, '-');
        for (i = 0; i < h->num_chunks; ++i)
        {
            if (h->extended[i])
            {
                char *group, *tag, type;
                int idx;
                size_t bytes;

                /* Get the tag data. */
                group = h->name_group[i];
                tag   = h->name_tag[i];
                type  = (char) (h->data_type[i]);
                idx   = h->user_index[i];
                bytes = h->payload_size_bytes[i];

                /* Display tag data. */
                oskar_log_message(log, p, -1, "[%3d] %-23s (%d bytes)",
                        i, oskar_mem_data_type_string(type), bytes);
                oskar_log_message(log, p, depth, "%s.%s : %d", group, tag, idx);
            }
        }
    }

    /* Release the handle. */
    oskar_binary_free(h);
}

#ifdef __cplusplus
}
#endif
