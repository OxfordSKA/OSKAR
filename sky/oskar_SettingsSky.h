/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_SETTINGS_SKY_NEW_H_
#define OSKAR_SETTINGS_SKY_NEW_H_

/**
 * @struct oskar_SettingsSkyFilter
 *
 * @brief Structure to hold sky model filter settings.
 *
 * @details
 * The structure holds parameters for a source filter.
 */
struct oskar_SettingsSkyFilter
{
    double flux_min;
    double flux_max;
    double radius_inner;
    double radius_outer;
};
typedef struct oskar_SettingsSkyFilter oskar_SettingsSkyFilter;

/**
 * @struct oskar_SettingsSkyGeneratorPowerLaw
 *
 * @brief Structure to hold settings for a sky model power-law generator.
 *
 * @details
 * The structure holds parameters for a sky model power-law generator.
 */
struct oskar_SettingsSkyGeneratorRandomPowerLaw
{
    oskar_SettingsSkyFilter filter;
    int num_sources;
    double flux_min;
    double flux_max;
    double power;
    int seed;
};
typedef struct oskar_SettingsSkyGeneratorRandomPowerLaw oskar_SettingsSkyGeneratorRandomPowerLaw;

/**
 * @struct oskar_SettingsSkyGeneratorRandomBrokenPowerLaw
 *
 * @brief Structure to hold settings for a random broken-power-law generator.
 *
 * @details
 * The structure holds parameters for a sky model random broken-power-law
 * generator.
 */
struct oskar_SettingsSkyGeneratorRandomBrokenPowerLaw
{
    oskar_SettingsSkyFilter filter;
    int num_sources;
    double flux_min;
    double flux_max;
    double threshold;
    double power1;
    double power2;
    int seed;
};
typedef struct oskar_SettingsSkyGeneratorRandomBrokenPowerLaw oskar_SettingsSkyGeneratorRandomBrokenPowerLaw;

/**
 * @struct oskar_SettingsSkyGeneratorHealpix
 *
 * @brief Structure to hold settings for a sky model HEALPix grid generator.
 *
 * @details
 * The structure holds parameters for a sky model HEALPix grid generator.
 */
struct oskar_SettingsSkyGeneratorHealpix
{
    oskar_SettingsSkyFilter filter;
    int nside;
};
typedef struct oskar_SettingsSkyGeneratorHealpix oskar_SettingsSkyGeneratorHealpix;

/**
 * @struct oskar_SettingsSkyGenerator
 *
 * @brief Structure to hold all sky model generator settings.
 *
 * @details
 * The structure holds parameters for all the sky model generators.
 */
struct oskar_SettingsSkyGenerator
{
    oskar_SettingsSkyGeneratorHealpix healpix;
    oskar_SettingsSkyGeneratorRandomPowerLaw random_power_law;
    oskar_SettingsSkyGeneratorRandomBrokenPowerLaw random_broken_power_law;
};
typedef struct oskar_SettingsSkyGenerator oskar_SettingsSkyGenerator;

enum
{
    OSKAR_NOISE_VLA_MEMO_146
};

/**
 * @struct oskar_SettingsSkyNoise
 *
 * @brief Structure to hold sky noise model settings.
 *
 * @details
 * The structure holds parameters for the sky noise model.
 */
struct oskar_SettingsSkyNoise
{
    int type; /**< Noise model type. */
    double spectral_index; /**< Frequency spectral index. */
    int seed;
};
typedef struct oskar_SettingsSkyNoise oskar_SettingsSkyNoise;

/**
 * @struct oskar_SettingsSky
 *
 * @brief Structure to hold sky model settings.
 *
 * @details
 * The structure holds parameters to construct a sky model.
 */
struct oskar_SettingsSkyNew
{
    char* input_sky_file;  /**< Name of sky model file to load. */
    oskar_SettingsSkyFilter input_sky_filter;
    char* gsm_file;        /**< Name of global sky model file to load. */
    oskar_SettingsSkyFilter gsm_filter;
    char* output_sky_file; /**< Optional name of output sky model. */
    oskar_SettingsSkyGenerator generator; /**< Generator parameters. */
    oskar_SettingsSkyNoise noise_model; /**< Noise model parameters. */
};
typedef struct oskar_SettingsSkyNew oskar_SettingsSkyNew;

#endif /* OSKAR_SETTINGS_SKY_NEW_H_ */
