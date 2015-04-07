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

#ifndef OSKAR_SETTINGS_BEAM_PATTERN_H_
#define OSKAR_SETTINGS_BEAM_PATTERN_H_

/**
 * @file oskar_SettingsBeamPattern.h
 */

/**
 * @brief
 * Describes the coordinates at which the beam pattern is evaluated.
 */
enum OSKAR_BEAM_PATTERN_COORDS
{
    OSKAR_BEAM_PATTERN_COORDS_UNDEF,
    OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE,
    OSKAR_BEAM_PATTERN_COORDS_HEALPIX,
    OSKAR_BEAM_PATTERN_COORDS_SKY_MODEL
};

/**
 * @brief
 * Describes the coordinate frame in which the beam pattern is evaluated.
 */
enum OSKAR_BEAM_PATTERN_FRAME
{
    OSKAR_BEAM_PATTERN_FRAME_UNDEF,
    OSKAR_BEAM_PATTERN_FRAME_EQUATORIAL,
    OSKAR_BEAM_PATTERN_FRAME_HORIZON
};

/**
 * @brief
 * Describes the option to average a single axis.
 */
enum OSKAR_BEAM_PATTERN_AVERAGE_SINGLE_AXIS
{
    OSKAR_BEAM_PATTERN_AVERAGE_NONE,
    OSKAR_BEAM_PATTERN_AVERAGE_TIME,
    OSKAR_BEAM_PATTERN_AVERAGE_CHANNEL
};

/**
 * @struct oskar_SettingsBeamPattern
 *
 * @brief Structure to hold beam pattern settings.
 *
 * @details
 * The structure holds parameters for making a beam pattern.
 */
struct oskar_SettingsBeamPattern
{
    int all_stations;
    int num_active_stations;
    int* station_ids;

    int coord_grid_type;  /* The type of beam pattern coordinates */
    int coord_frame_type; /* Coordinate frame for beam pattern evaluation */

    /* Beam pattern image settings */
    double fov_deg[2];
    int size[2];

    int healpix_coord_type;
    int nside;

    int horizon_clip;   /* Bool, to toggle horizon clip of the beam pattern */

    char* sky_model;

    /* Averaging options. */
    int separate_time_and_channel;
    int average_time_and_channel;
    int average_single_axis;

    /* Output file options. */
    char* root_path;
    int station_text_raw_complex;
    int station_text_amp;
    int station_text_phase;
    int station_text_auto_power_stokes_i;
    int station_text_ixr;
    int station_fits_amp;
    int station_fits_phase;
    int station_fits_auto_power_stokes_i;
    int station_fits_ixr;
    int telescope_text_cross_power_stokes_i_raw_complex;
    int telescope_text_cross_power_stokes_i_amp;
    int telescope_text_cross_power_stokes_i_phase;
    int telescope_fits_cross_power_stokes_i_amp;
    int telescope_fits_cross_power_stokes_i_phase;
};
typedef struct oskar_SettingsBeamPattern oskar_SettingsBeamPattern;

#endif /* OSKAR_SETTINGS_BEAM_PATTERN_H_ */
