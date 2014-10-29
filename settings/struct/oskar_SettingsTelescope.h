/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#ifndef OSKAR_SETTINGS_TELESCOPE_H_
#define OSKAR_SETTINGS_TELESCOPE_H_

/**
 * @file oskar_SettingsTelescope.h
 */

/**
 * @struct oskar_SettingsArrayElement
 *
 * @brief Structure to hold station element settings.
 *
 * @details
 * The structure holds station element parameters that can be used to override
 * those in the station files.
 */
struct oskar_SettingsArrayElement
{
    int apodisation_type;
    double gain;
    double gain_error_fixed;
    double gain_error_time;
    double phase_error_fixed_rad;
    double phase_error_time_rad;
    double position_error_xy_m;
    double x_orientation_error_rad;
    double y_orientation_error_rad;

    /* Random seeds. */
    int seed_gain_errors;
    int seed_phase_errors;
    int seed_time_variable_errors;
    int seed_position_xy_errors;
    int seed_x_orientation_error;
    int seed_y_orientation_error;
};
typedef struct oskar_SettingsArrayElement oskar_SettingsArrayElement;

/**
 * @struct oskar_SettingsArrayPattern
 *
 * @brief Structure to hold settings for the station array pattern evaluation.
 *
 * @details
 * The structure holds settings for the station array pattern evaluation.
 */
struct oskar_SettingsArrayPattern
{
    int enable;
    int normalise;
    oskar_SettingsArrayElement element;
};
typedef struct oskar_SettingsArrayPattern oskar_SettingsArrayPattern;

/**
 * @struct oskar_SettingsElementTaper
 *
 * @brief Structure to hold settings for the station element tapering.
 *
 * @details
 * The structure holds settings for the station element tapering.
 */
struct oskar_SettingsElementTaper
{
    int type;
    double cosine_power;
    double gaussian_fwhm_rad;
};
typedef struct oskar_SettingsElementTaper oskar_SettingsElementTaper;

/**
 * @struct oskar_SettingsElementPattern
 *
 * @brief Structure to hold settings for the station element pattern evaluation.
 *
 * @details
 * The structure holds settings for the station element pattern evaluation.
 */
struct oskar_SettingsElementPattern
{
    int enable_numerical_patterns;
    int functional_type;
    int dipole_length_units;
    double dipole_length;
    oskar_SettingsElementTaper taper;
};
typedef struct oskar_SettingsElementPattern oskar_SettingsElementPattern;

/**
 * @struct oskar_SettingsApertureArray
 *
 * @brief Structure to hold settings for aperture array stations.
 *
 * @details
 * The structure holds settings for aperture array stations.
 */
struct oskar_SettingsApertureArray
{
    oskar_SettingsArrayPattern array_pattern;
    oskar_SettingsElementPattern element_pattern;
};
typedef struct oskar_SettingsApertureArray oskar_SettingsApertureArray;

/**
 * @struct oskar_SettingsGaussianBeam
 *
 * @brief Structure to hold settings for stations with a Gaussian beam.
 *
 * @details
 * The structure holds settings for stations with a Gaussian beam.
 */
struct oskar_SettingsGaussianBeam
{
    double fwhm_deg;
    double ref_freq_hz;
};
typedef struct oskar_SettingsGaussianBeam oskar_SettingsGaussianBeam;

/**
 * @struct oskar_SettingsTelescope
 *
 * @brief Structure to hold telescope model settings.
 *
 * @details
 * The structure holds telescope model parameters.
 */
struct oskar_SettingsTelescope
{
    char* input_directory;
    char* output_directory;
    double longitude_rad;
    double latitude_rad;
    double altitude_m;
    int station_type;
    int normalise_beams_at_phase_centre;
    int pol_mode;
    int allow_station_beam_duplication;
    oskar_SettingsApertureArray aperture_array;
    oskar_SettingsGaussianBeam gaussian_beam;
};
typedef struct oskar_SettingsTelescope oskar_SettingsTelescope;

#endif /* OSKAR_SETTINGS_TELESCOPE_H_ */
