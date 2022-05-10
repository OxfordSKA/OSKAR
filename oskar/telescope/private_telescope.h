/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_TELESCOPE_H_
#define OSKAR_PRIVATE_TELESCOPE_H_

#include <telescope/station/oskar_station.h>
#include <gains/oskar_gains.h>
#include <harp/oskar_harp.h>

struct oskar_Telescope
{
    /* Private structure meta-data. */
    int precision;
    int mem_location;

    /* Properties that don't depend on stations. */
    int pol_mode;                /* Polarisation mode (full or scalar). */
    double lon_rad;              /* Geodetic longitude of telescope, in radians. */
    double lat_rad;              /* Geodetic latitude of telescope, in radians. */
    double alt_metres;           /* Altitude of telescope above ellipsoid, in metres. */
    double pm_x_rad;             /* Polar motion (x-component) in radians. */
    double pm_y_rad;             /* Polar motion (y-component) in radians. */
    int phase_centre_coord_type; /* Phase centre coordinate type. */
    double phase_centre_rad[2];  /* Longitude and latitude of phase centre, in radians. */
    double channel_bandwidth_hz; /* Channel bandwidth, in Hz. */
    double time_average_sec;     /* Time average smearing duration, in sec. */
    double uv_filter_min;        /* Minimum allowed UV distance. */
    double uv_filter_max;        /* Maximum allowed UV distance. */
    int uv_filter_units;         /* Unit of allowed UV distance (OSKAR_METRES or OSKAR_WAVELENGTHS). */
    int noise_enabled;           /* Flag set if thermal noise is enabled. */
    unsigned int noise_seed;     /* Random generator seed. */

    /* Gain model. */
    oskar_Gains* gains;

    /* HARP data. */
    int harp_num_freq;
    oskar_Mem* harp_freq_cpu;
    oskar_Harp** harp_data;

    /* Ionosphere parameters. */
    int ionosphere_screen_type;
    oskar_Mem* tec_screen_path;
    double tec_screen_height_km;
    double tec_screen_pixel_size_m;
    double tec_screen_time_interval_sec;
    int isoplanatic_screen;

    /* Station data. */
    int supplied_coord_type;                           /* Type of coordinates specified in telescope model. */
    int num_stations;                                  /* Number of stations in the model. */
    int num_station_models;                            /* Number of station models (size of station array). */
    oskar_Station** station;                           /* Array of station structure handles. */
    oskar_Mem* station_type_map;                       /* Integer array of station type indices. */
    oskar_Mem* station_true_geodetic_rad[3];           /* True geodetic station coordinates, in radians/metres (altitude) - always double precision, CPU. */
    oskar_Mem* station_true_offset_ecef_metres[3];     /* True station coordinates, in metres (offset ECEF). */
    oskar_Mem* station_true_enu_metres[3];             /* True station coordinate, in metres (horizon). */
    oskar_Mem* station_measured_offset_ecef_metres[3]; /* Measured station coordinates, in metres (offset ECEF). */
    oskar_Mem* station_measured_enu_metres[3];         /* Measured station coordinate, in metres (horizon). */
    int max_station_size;                              /* Maximum station size (number of elements) */
    int max_station_depth;                             /* Maximum station depth. */
    int allow_station_beam_duplication;                /* True if station beam duplication is allowed. */
    int enable_numerical_patterns;                     /* True if numerical element patterns are enabled. */
};

#ifndef OSKAR_TELESCOPE_TYPEDEF_
#define OSKAR_TELESCOPE_TYPEDEF_
typedef struct oskar_Telescope oskar_Telescope;
#endif /* OSKAR_TELESCOPE_TYPEDEF_ */

#endif /* include guard */
