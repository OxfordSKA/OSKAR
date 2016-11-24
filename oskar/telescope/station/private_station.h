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

#ifndef OSKAR_PRIVATE_STATION_H_
#define OSKAR_PRIVATE_STATION_H_

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
    int normalise_array_pattern;  /* True if the station beam should be normalised by the number of antennas. */
    int enable_array_pattern;     /* True if the array factor should be evaluated. */
    int common_element_orientation; /* True if elements share a common orientation (auto determined). */
    int array_is_3d;              /* True if array is 3-dimensional (auto determined; default false). */
    int apply_element_errors;     /* True if element gain and phase errors should be applied (auto determined; default false). */
    int apply_element_weight;     /* True if weights should be modified by user-supplied complex beamforming weights (auto determined; default false). */
    unsigned int seed_time_variable_errors;   /* Seed for time variable errors. */
    oskar_Mem* element_true_x_enu_metres;     /* True horizon element x-coordinates, in metres, towards East. */
    oskar_Mem* element_true_y_enu_metres;     /* True horizon element y-coordinates, in metres, towards North. */
    oskar_Mem* element_true_z_enu_metres;     /* True horizon element z-coordinates, in metres, towards the zenith. */
    oskar_Mem* element_measured_x_enu_metres; /* Measured horizon element x-coordinates, in metres, towards East. */
    oskar_Mem* element_measured_y_enu_metres; /* Measured horizon element y-coordinates, in metres, towards North. */
    oskar_Mem* element_measured_z_enu_metres; /* Measured horizon element z-coordinates, in metres, towards the zenith. */
    oskar_Mem* element_gain;                  /* Per-element gain factor (default 1.0) */
    oskar_Mem* element_gain_error;            /* Standard deviation of per-element time-variable gain factor (default 0.0) */
    oskar_Mem* element_phase_offset_rad;      /* Per-element systematic phase offset, in radians (default 0.0) */
    oskar_Mem* element_phase_error_rad;       /* Standard deviation of per-element time-variable phase, in radians (default 0.0) */
    oskar_Mem* element_weight;                /* Element complex weight (set to 1.0, 0.0 unless using apodisation). */
    oskar_Mem* element_types;     /* Integer array of element types (default 0). */
    oskar_Mem* element_types_cpu; /* Integer array of element types guaranteed to be in CPU memory (default 0). */
    oskar_Mem* element_mount_types_cpu; /* Char array of element mount types guaranteed to be in CPU memory. */
    oskar_Mem* element_x_alpha_cpu; /* X element Euler angle orientation, guaranteed to be in CPU memory. */
    oskar_Mem* element_x_beta_cpu;  /* X element Euler angle orientation, guaranteed to be in CPU memory. */
    oskar_Mem* element_x_gamma_cpu; /* X element Euler angle orientation, guaranteed to be in CPU memory. */
    oskar_Mem* element_y_alpha_cpu; /* Y element Euler angle orientation, guaranteed to be in CPU memory. */
    oskar_Mem* element_y_beta_cpu;  /* Y element Euler angle orientation, guaranteed to be in CPU memory. */
    oskar_Mem* element_y_gamma_cpu; /* Y element Euler angle orientation, guaranteed to be in CPU memory. */
    oskar_Station** child;        /* Array of child station handles (pointer is NULL if none). */
    oskar_Element** element;      /* Array of element models per element type (pointer is NULL if there are child stations). */

    /* Data used only for aperture array stations with fixed beams. */
    int num_permitted_beams;
    oskar_Mem* permitted_beam_az_rad;
    oskar_Mem* permitted_beam_el_rad;
};

#endif /* OSKAR_PRIVATE_STATION_H_ */
