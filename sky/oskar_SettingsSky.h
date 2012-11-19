/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_SETTINGS_SKY_H_
#define OSKAR_SETTINGS_SKY_H_

/**
 * @file oskar_SettingsSky.h
 */

#include "oskar_global.h"

/**
 * @struct oskar_SettingsSkyFilter
 *
 * @brief Structure to hold sky model filter settings.
 *
 * @details
 * The structure holds parameters for a source filter.
 */
struct OSKAR_EXPORT oskar_SettingsSkyFilter
{
    double flux_min;
    double flux_max;
    double radius_inner;
    double radius_outer;
};
typedef struct oskar_SettingsSkyFilter oskar_SettingsSkyFilter;

/**
 * @struct oskar_SettingsSkyExtendedSources
 *
 * @brief Holds extended source settings which apply to all sources.
 */
struct OSKAR_EXPORT oskar_SettingsSkyExtendedSources
{
    double FWHM_major;      /**< Major axis FWHM in radians. */
    double FWHM_minor;      /**< Minor axis FWHM in radians. */
    double position_angle;  /**< Position angle in radians. */
};
typedef struct oskar_SettingsSkyExtendedSources oskar_SettingsSkyExtendedSources;

/**
 * @struct oskar_SettingsSkyFits
 *
 * @brief Holds FITS file import parameters.
 */
struct OSKAR_EXPORT oskar_SettingsSkyFits
{
    double spectral_index;    /**< Spectral index value of pixels in file. */
    double noise_floor;       /**< Noise floor in Jy. */
    double min_peak_fraction; /**< Minimum fraction of peak value to accept. */
    int downsample_factor;    /**< Factor by which to downsample pixel grid. */
};
typedef struct oskar_SettingsSkyFits oskar_SettingsSkyFits;

/**
 * @struct oskar_SettingsSkyHealpixFits
 *
 * @brief Holds FITS file import parameters.
 */
struct OSKAR_EXPORT oskar_SettingsSkyHealpixFits
{
    int num_files;  /**< Number of HEALPix-FITS files to load. */
    char** file;    /**< List of HEALPix-FITS input sky model files. */
    int coord_sys;  /**< Coordinate system to apply (enumerator). */
    int map_units;  /**< Units of input map (enumerator). */
    oskar_SettingsSkyFilter filter;
    oskar_SettingsSkyExtendedSources extended_sources;
};
typedef struct oskar_SettingsSkyHealpixFits oskar_SettingsSkyHealpixFits;

enum {
    OSKAR_COORD_SYS_GALACTIC,
    OSKAR_COORD_SYS_EQUATORIAL
};

enum {
    OSKAR_MAP_UNITS_JY,
    OSKAR_MAP_UNITS_K_PER_SR,
    OSKAR_MAP_UNITS_MK_PER_SR
};

/**
 * @struct oskar_SettingsSkyGeneratorPowerLaw
 *
 * @brief Structure to hold settings for a sky model power-law generator.
 *
 * @details
 * The structure holds parameters for a sky model power-law generator.
 */
struct OSKAR_EXPORT oskar_SettingsSkyGeneratorRandomPowerLaw
{
    oskar_SettingsSkyFilter filter;
    oskar_SettingsSkyExtendedSources extended_sources;
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
struct OSKAR_EXPORT oskar_SettingsSkyGeneratorRandomBrokenPowerLaw
{
    oskar_SettingsSkyFilter filter;
    oskar_SettingsSkyExtendedSources extended_sources;
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
struct OSKAR_EXPORT oskar_SettingsSkyGeneratorHealpix
{
    oskar_SettingsSkyFilter filter;
    oskar_SettingsSkyExtendedSources extended_sources;
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
struct OSKAR_EXPORT oskar_SettingsSkyGenerator
{
    oskar_SettingsSkyGeneratorHealpix healpix;
    oskar_SettingsSkyGeneratorRandomPowerLaw random_power_law;
    oskar_SettingsSkyGeneratorRandomBrokenPowerLaw random_broken_power_law;
};
typedef struct oskar_SettingsSkyGenerator oskar_SettingsSkyGenerator;

/**
 * @struct oskar_SettingsSkySpectralIndex
 *
 * @brief Structure to hold all sky model spectral index overrides.
 *
 * @details
 * The structure holds parameters for the sky model spectral index overrides.
 */
struct OSKAR_EXPORT oskar_SettingsSkySpectralIndex
{
    int override;
    double ref_frequency_hz;
    double mean;
    double std_dev;
    int seed;
};
typedef struct oskar_SettingsSkySpectralIndex oskar_SettingsSkySpectralIndex;

/**
 * @struct oskar_SettingsSky
 *
 * @brief Structure to hold sky model settings.
 *
 * @details
 * The structure holds parameters to construct a sky model.
 */
struct OSKAR_EXPORT oskar_SettingsSky
{
    int num_sky_files;     /**< Number of sky model files to load. */
    char** input_sky_file; /**< List of sky model files to load. */
    oskar_SettingsSkyFilter input_sky_filter;
    oskar_SettingsSkyExtendedSources input_sky_extended_sources;

    char* gsm_file;        /**< Name of global sky model file to load. */
    oskar_SettingsSkyFilter gsm_filter;
    oskar_SettingsSkyExtendedSources gsm_extended_sources;

    char* output_text_file; /**< Optional name of output sky model text file. */
    char* output_binary_file; /**< Optional name of output sky model binary file. */
    oskar_SettingsSkyGenerator generator; /**< Generator parameters. */

    oskar_SettingsSkySpectralIndex spectral_index; /**< Spectral index overrides. */

    int num_fits_files;    /**< Number of FITS files to load. */
    char** fits_file;      /**< List of FITS input sky model files. */
    oskar_SettingsSkyFits fits_file_settings;

    oskar_SettingsSkyHealpixFits healpix_fits;
};
typedef struct oskar_SettingsSky oskar_SettingsSky;

#endif /* OSKAR_SETTINGS_SKY_H_ */
