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

#ifndef OSKAR_SETTINGS_SKY_H_
#define OSKAR_SETTINGS_SKY_H_

/**
 * @file oskar_SettingsSky.h
 */

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
    double radius_inner_rad;
    double radius_outer_rad;
};
typedef struct oskar_SettingsSkyFilter oskar_SettingsSkyFilter;

/**
 * @struct oskar_SettingsSkyExtendedSources
 *
 * @brief Holds extended source settings which apply to all sources.
 */
struct oskar_SettingsSkyExtendedSources
{
    double FWHM_major_rad;      /**< Major axis FWHM in radians. */
    double FWHM_minor_rad;      /**< Minor axis FWHM in radians. */
    double position_angle_rad;  /**< Position angle in radians. */
};
typedef struct oskar_SettingsSkyExtendedSources oskar_SettingsSkyExtendedSources;

/**
 * @struct oskar_SettingsSkyPolarisation
 *
 * @brief Holds polarisation settings for sky model generators.
 */
struct oskar_SettingsSkyPolarisation
{
    double mean_pol_fraction; /**< Mean polarisation fraction. */
    double std_pol_fraction;  /**< Standard deviation of polarisation fraction. */
    double mean_pol_angle_rad; /**< Mean polarisation angle in radians. */
    double std_pol_angle_rad; /**< Standard deviation of polarisation angle in radians. */
    int seed;
};
typedef struct oskar_SettingsSkyPolarisation oskar_SettingsSkyPolarisation;

/**
 * @struct oskar_SettingsSkyOskar
 *
 * @brief Holds OSKAR sky model import parameters.
 */
struct oskar_SettingsSkyOskar
{
    int num_files;  /**< Number of OSKAR sky model files to load. */
    char** file;    /**< List of OSKAR sky model files. */
    oskar_SettingsSkyFilter filter;
    oskar_SettingsSkyExtendedSources extended_sources;
};
typedef struct oskar_SettingsSkyOskar oskar_SettingsSkyOskar;

/**
 * @struct oskar_SettingsSkyGsm
 *
 * @brief Holds GSM file import parameters.
 */
struct oskar_SettingsSkyGsm
{
    char* file;    /**< Filename of GSM file to load. */
    oskar_SettingsSkyFilter filter;
    oskar_SettingsSkyExtendedSources extended_sources;
};
typedef struct oskar_SettingsSkyGsm oskar_SettingsSkyGsm;

/**
 * @struct oskar_SettingsSkyFitsImage
 *
 * @brief Holds FITS image file import parameters.
 */
struct oskar_SettingsSkyFitsImage
{
    int num_files;            /**< Number of FITS image files to load. */
    char** file;              /**< List of FITS image files. */
    double spectral_index;    /**< Spectral index value of pixels in file. */
    double noise_floor;       /**< Noise floor in Jy. */
    double min_peak_fraction; /**< Minimum fraction of peak value to accept. */
    int downsample_factor;    /**< Factor by which to downsample pixel grid. */
};
typedef struct oskar_SettingsSkyFitsImage oskar_SettingsSkyFitsImage;

/**
 * @struct oskar_SettingsSkyHealpixFits
 *
 * @brief Holds FITS file import parameters.
 */
struct oskar_SettingsSkyHealpixFits
{
    int num_files;  /**< Number of HEALPix-FITS files to load. */
    char** file;    /**< List of HEALPix-FITS input sky model files. */
    int coord_sys;  /**< Coordinate system to apply (enumerator). */
    int map_units;  /**< Units of input map (enumerator). */
    oskar_SettingsSkyFilter filter;
    oskar_SettingsSkyExtendedSources extended_sources;
};
typedef struct oskar_SettingsSkyHealpixFits oskar_SettingsSkyHealpixFits;

enum OSKAR_MAP_UNITS
{
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
struct oskar_SettingsSkyGeneratorRandomPowerLaw
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
struct oskar_SettingsSkyGeneratorRandomBrokenPowerLaw
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
struct oskar_SettingsSkyGeneratorHealpix
{
    oskar_SettingsSkyFilter filter;
    oskar_SettingsSkyExtendedSources extended_sources;
    int nside;
    double amplitude;
};
typedef struct oskar_SettingsSkyGeneratorHealpix oskar_SettingsSkyGeneratorHealpix;

/**
 * @struct oskar_SettingsSkyGeneratorGrid
 *
 * @brief Structure to hold settings for a sky model grid generator.
 *
 * @details
 * The structure holds parameters for a sky model grid generator.
 */
struct oskar_SettingsSkyGeneratorGrid
{
    oskar_SettingsSkyExtendedSources extended_sources;
    oskar_SettingsSkyPolarisation pol;
    int side_length;
    double fov_rad;
    double mean_flux_jy;
    double std_flux_jy;
    int seed;
};
typedef struct oskar_SettingsSkyGeneratorGrid oskar_SettingsSkyGeneratorGrid;

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
    oskar_SettingsSkyGeneratorGrid grid;
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
struct oskar_SettingsSkySpectralIndex
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
struct oskar_SettingsSky
{
    oskar_SettingsSkyOskar oskar_sky_model;
    oskar_SettingsSkyGsm gsm;
    oskar_SettingsSkyFitsImage fits_image;
    oskar_SettingsSkyHealpixFits healpix_fits;
    oskar_SettingsSkyGenerator generator; /**< All generator parameters. */
    oskar_SettingsSkySpectralIndex spectral_index; /**< Spectral index overrides. */
    char* output_text_file; /**< Optional name of output sky model text file. */
    char* output_binary_file; /**< Optional name of output sky model binary file. */
    double common_flux_filter_min_jy;
    double common_flux_filter_max_jy;
    int zero_failed_gaussians; /**< Zero (remove) sources with failed Gaussian width solutions. */
    int apply_horizon_clip;
};
typedef struct oskar_SettingsSky oskar_SettingsSky;

#endif /* OSKAR_SETTINGS_SKY_H_ */
