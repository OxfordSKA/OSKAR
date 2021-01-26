/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_VIS_HEADER_H_
#define OSKAR_PRIVATE_VIS_HEADER_H_

#include <mem/oskar_mem.h>

/*
 * Holds visibility header data, including station coordinates.
 * Arrays here will always be in CPU memory.
 */
struct oskar_VisHeader
{
    oskar_Mem* settings;             /* Settings file contents. */
    oskar_Mem* telescope_path;       /* Path to telescope model. */

    int num_tags_header;             /* Number of tags read from the header. */
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

    oskar_Mem* station_offset_ecef_metres[3]; /* Station coordinates [m] (offset ECEF). */
    oskar_Mem** element_enu_metres[3]; /* Length num_stations. */
};

#ifndef OSKAR_VIS_HEADER_TYPEDEF_
#define OSKAR_VIS_HEADER_TYPEDEF_
typedef struct oskar_VisHeader oskar_VisHeader;
#endif /* OSKAR_VIS_HEADER_TYPEDEF_ */

#endif /* include guard */
