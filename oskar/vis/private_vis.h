/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#ifndef OSKAR_PRIVATE_VIS_H_
#define OSKAR_PRIVATE_VIS_H_

#include <mem/oskar_mem.h>

/*
 * Deprecated structure to hold visibility data.
 *
 * Visibility data consists of a set of baseline (u,v,w) coordinates and
 * one or more complex visibility amplitudes.
 *
 * Baseline coordinates are stored as three separate arrays of length corresponding
 * to the number of visibility samples in the data set. Coordinates are real
 * valued and can be either double or single precision. Coordinates are ordered
 * by channel ID, time and baseline where channel is the slowest varying dimension
 * and baseline is the fastest varying.
 *
 * Axis order: channel(slowest) -> time -> baseline (fastest)
 *
 * Visibility amplitudes are sorted as an array of complex scalar or 2-by-2 matrix
 * elements which can be either double or single precision according to the
 * amplitude type ID. The dimension ordering is the same as that used for
 * the baseline coordinate arrays. The use of scalar or matrix elements in the
 * array determines the number of polarisations stored in the data. Scalar elements
 * represent a single polarisation axis whereas matrix elements store 4 polarisations
 * per data sample.
 */
struct oskar_Vis
{
    oskar_Mem* settings_path;    /* Path to settings file. */
    oskar_Mem* telescope_path;   /* Path to telescope model. */
    oskar_Mem* settings;         /* Settings file contents. */

    int num_channels;            /* Number of frequency channels. */
    int num_times;               /* Number of time samples. */
    int num_stations;            /* Number of interferometer stations. */
    int num_baselines;           /* Number of interferometer baselines. */

    double freq_start_hz;        /* Start frequency, in Hz. */
    double freq_inc_hz;          /* Frequency increment, in Hz. */
    double channel_bandwidth_hz; /* Frequency channel bandwidth, in Hz */
    double time_start_mjd_utc;   /* Start time in MJD, UTC. */
    double time_inc_sec;         /* Time increment, in seconds. */
    double time_average_sec;     /* Time average smearing duration, in seconds. */
    double phase_centre_ra_deg;  /* Pointing phase centre RA, in degrees. */
    double phase_centre_dec_deg; /* Pointing phase centre Dec, in degrees. */
    double telescope_lon_deg;    /* Reference longitude of the telescope, in degrees */
    double telescope_lat_deg;    /* Reference latitude of the telescope, in degrees */
    double telescope_alt_metres; /* Reference altitude of the telescope, in metres. */

    oskar_Mem* station_x_offset_ecef_metres; /* Station x-coordinate, in metres (offset ECEF). */
    oskar_Mem* station_y_offset_ecef_metres; /* Station y-coordinate, in metres (offset ECEF). */
    oskar_Mem* station_z_offset_ecef_metres; /* Station z-coordinate, in metres (offset ECEF). */
    oskar_Mem* baseline_uu_metres;           /* Baseline coordinates, in metres. */
    oskar_Mem* baseline_vv_metres;           /* Baseline coordinates, in metres. */
    oskar_Mem* baseline_ww_metres;           /* Baseline coordinates, in metres. */
    oskar_Mem* amplitude;                    /* Complex visibility amplitude. */
};

#ifndef OSKAR_VIS_TYPEDEF_
#define OSKAR_VIS_TYPEDEF_
typedef struct oskar_Vis oskar_Vis;
#endif /* OSKAR_VIS_TYPEDEF_ */

#endif /* OSKAR_PRIVATE_VIS_H_ */
