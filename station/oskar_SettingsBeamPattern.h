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

#ifndef OSKAR_SETTINGS_BEAM_PATTERN_H_
#define OSKAR_SETTINGS_BEAM_PATTERN_H_

/**
 * @file oskar_SettingsBeamPattern.h
 */


/**
 * @brief
 * Enum describing the coordinates at which the beam pattern is evaluated.
 */
enum {
    OSKAR_BEAM_PATTERN_COORDS_UNDEF,
    OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE,
    OSKAR_BEAM_PATTERN_COORDS_HEALPIX
};

/**
 * @brief
 * Enum describing the coordinates frame for which the beam pattern is evaluated.
 */
enum {
    OSKAR_BEAM_PATTERN_FRAME_UNDEF,
    OSKAR_BEAM_PATTERN_FRAME_EQUATORIAL,
    OSKAR_BEAM_PATTERN_FRAME_HORIZON
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
    int station_id;

    int coord_grid_type;  /* The type of beam pattern coordinates */
    int coord_frame_type; /* Coordinate frame for beam pattern evaluation */

    /* Beam pattern image settings */
    double fov_deg[2];
    int size[2];

    int healpix_coord_type;
    int nside;

    int horizon_clip;   /* Bool, to toggle horizon clip of the beam pattern */

    char* oskar_image_voltage;
    char* oskar_image_phase;
    char* oskar_image_complex;
    char* oskar_image_total_intensity;
    char* fits_image_voltage;
    char* fits_image_phase;
    char* fits_image_total_intensity;
};
typedef struct oskar_SettingsBeamPattern oskar_SettingsBeamPattern;

#endif /* OSKAR_SETTINGS_BEAM_PATTERN_H_ */
