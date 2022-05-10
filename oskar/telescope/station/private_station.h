/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_STATION_H_
#define OSKAR_PRIVATE_STATION_H_

#include <gains/oskar_gains.h>
#include <harp/oskar_harp.h>
#include <mem/oskar_mem.h>
#include <telescope/station/element/oskar_element.h>

/* Forward declaration. */
struct oskar_Station;

#ifndef OSKAR_STATION_TYPEDEF_
#define OSKAR_STATION_TYPEDEF_
typedef struct oskar_Station oskar_Station;
#endif /* OSKAR_STATION_TYPEDEF_ */

struct oskar_Station
{
    /* Private structure meta-data. */
    int unique_id;                /* Unique ID for station within telescope. */
    int precision;                /* Numerical precision of most arrays. */
    int mem_location;             /* Memory location of most arrays. */

    /* Data common to all station types -------------------------------------*/
    int station_type;             /* Type of the station (enumerator). */
    int normalise_final_beam;     /* Flag to specify whether the station beam should be completely normalised. */
    double offset_ecef[3];        /* Offset ECEF coordinates of the station, in metres. */
    double lon_rad;               /* Geodetic east longitude of station, in radians. */
    double lat_rad;               /* Geodetic latitude of station, in radians. */
    double alt_metres;            /* Altitude of station above ellipsoid, in metres. */
    double pm_x_rad;              /* Polar motion (x-component) in radians. */
    double pm_y_rad;              /* Polar motion (y-component) in radians. */
    double beam_lon_rad;          /* Longitude of beam phase centre, in radians. */
    double beam_lat_rad;          /* Latitude of beam phase centre, in radians. */
    int beam_coord_type;          /* Enumerator describing beam spherical coordinate type (from oskar_global.h). */
    oskar_Mem* noise_freq_hz;     /* Frequency values, in Hz, at which noise RMS values are defined. */
    oskar_Mem* noise_rms_jy;      /* RMS noise values, in Jy, as a function of frequency. */

    /* Data used only for Gaussian beam stations  ---------------------------*/
    double gaussian_beam_fwhm_rad;   /* FWHM of Gaussian station beam, in degrees. */
    double gaussian_beam_reference_freq_hz; /* Reference frequency of the FHWM, in Hz. */

    /* Data used only for aperture array stations ---------------------------*/
    int identical_children;       /* True if all child stations are identical. */
    int num_elements;             /* Number of antenna elements in the station (auto determined). */
    int num_element_types;        /* Number of element types (this is the size of element_pattern array). */
    int normalise_array_pattern;  /* True if the array pattern should be normalised by the number of antennas. */
    int normalise_element_pattern;/* True if the element patterns should be normalised. */
    int enable_array_pattern;     /* True if the array factor should be evaluated. */
    int common_element_orientation; /* True if elements share a common orientation (auto determined). */
    int common_pol_beams;         /* True if beams for both polarisations can be formed in the same way (auto determined). */
    int swap_xy;                  /* True if the X and Y antennas should be swapped in the output. */
    int array_is_3d;              /* True if array is 3-dimensional (auto determined; default false). */
    int apply_element_errors;     /* True if element gain and phase errors should be applied (auto determined; default false). */
    int apply_element_weight;     /* True if weights should be modified by user-supplied complex beamforming weights (auto determined; default false). */
    unsigned int seed_time_variable_errors;       /* Seed for time variable errors. */
    oskar_Mem* element_true_enu_metres[2][3];     /* True horizon element ENU coordinates, in metres. */
    oskar_Mem* element_measured_enu_metres[2][3]; /* Measured horizon element ENU coordinates, in metres. */
    oskar_Mem* element_gain[2];                   /* Element gain factor (default 1.0) */
    oskar_Mem* element_gain_error[2];             /* Standard deviation of per-element time-variable gain factor (default 0.0) */
    oskar_Mem* element_phase_offset_rad[2];       /* Element systematic phase offset, in radians (default 0.0) */
    oskar_Mem* element_phase_error_rad[2];        /* Standard deviation of per-element time-variable phase, in radians (default 0.0) */
    oskar_Mem* element_weight[2];                 /* Element complex weight (set to 1.0, 0.0 unless using apodisation). */
    oskar_Mem* element_cable_length_error[2];     /* Element cable length error, in metres. */
    oskar_Mem* element_euler_cpu[2][3];           /* Element Euler angles, guaranteed to be in CPU memory. */
    oskar_Mem* element_types;     /* Integer array of element types (default 0). */
    oskar_Mem* element_types_cpu; /* Integer array of element types guaranteed to be in CPU memory (default 0). */
    oskar_Mem* element_mount_types_cpu; /* Char array of element mount types guaranteed to be in CPU memory. */
    oskar_Station** child;        /* Array of child station handles (pointer is NULL if none). */
    oskar_Element** element;      /* Array of element models per element type (pointer is NULL if there are child stations). */

    /* Gain model. */
    oskar_Gains* gains;

    /* HARP data. */
    int harp_num_freq;
    oskar_Mem* harp_freq_cpu;
    oskar_Harp** harp_data;

    /* Data used only for aperture array stations with fixed beams. */
    int num_permitted_beams;
    oskar_Mem* permitted_beam_az_rad;
    oskar_Mem* permitted_beam_el_rad;
};

#endif /* OSKAR_PRIVATE_STATION_H_ */
