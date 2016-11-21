/*
 * Copyright (c) 2015, The University of Oxford
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

#ifndef OSKAR_PRIVATE_VIS_HEADER_H_
#define OSKAR_PRIVATE_VIS_HEADER_H_

#include <oskar_mem.h>

/*
 * Holds visibility header data, including station coordinates.
 * Arrays here will always be in CPU memory.
 */
struct oskar_VisHeader
{
    oskar_Mem* settings;             /* Settings file contents. */
    oskar_Mem* telescope_path;       /* Path to telescope model. */

    int num_tags_per_block;          /* Number of tags per block in binary file. */
    int write_autocorr;              /* True if auto-correlations are written. */
    int write_crosscorr;             /* True if cross-correlations are written. */
    int amp_type;                    /* Memory type of the amplitudes. */
    int coord_precision;             /* Memory type of the coordinates. */
    int max_times_per_block;         /* Maximum no. time samples per block. */
    int num_times_total;             /* Total no. time samples. */
    int max_channels_per_block;      /* Maximum no. channels per block. */
    int num_channels_total;          /* Total no. channels. */
    int num_stations;                /* No. interferometer stations. */
    int pol_type;                    /* Polarisation type enumerator. */

    int phase_centre_type;           /* Phase centre coordinate type. */
    double phase_centre_deg[2];      /* Phase centre coordinates [deg]. */
    double freq_start_hz;            /* Start frequency [Hz]. */
    double freq_inc_hz;              /* Frequency increment [Hz]. */
    double channel_bandwidth_hz;     /* Frequency channel bandwidth [Hz]. */
    double time_start_mjd_utc;       /* Start time [MJD(UTC)]. */
    double time_inc_sec;             /* Smallest time increment [s]. */
    double time_average_sec;         /* Time average smearing duration [s]. */
    double telescope_centre_lon_deg; /* Telescope reference longitude [deg]. */
    double telescope_centre_lat_deg; /* Telescope reference latitude [deg]. */
    double telescope_centre_alt_m;   /* Telescope reference altitude [m]. */

    oskar_Mem* station_x_offset_ecef_metres; /* Station x-coordinate, in metres (offset ECEF). */
    oskar_Mem* station_y_offset_ecef_metres; /* Station y-coordinate, in metres (offset ECEF). */
    oskar_Mem* station_z_offset_ecef_metres; /* Station z-coordinate, in metres (offset ECEF). */
};

#ifndef OSKAR_VIS_HEADER_TYPEDEF_
#define OSKAR_VIS_HEADER_TYPEDEF_
typedef struct oskar_VisHeader oskar_VisHeader;
#endif /* OSKAR_VIS_HEADER_TYPEDEF_ */

#endif /* OSKAR_PRIVATE_VIS_HEADER_H_ */
