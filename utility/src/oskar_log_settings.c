/*
 * Copyright (c) 2013, The University of Oxford
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

#include "utility/oskar_log_message.h"
#include "utility/oskar_log_value.h"
#include "utility/oskar_log_settings.h"
#include "station/oskar_StationModel.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define R2D 180.0/M_PI

#ifdef __cplusplus
extern "C" {
#endif


/* Column width for key prefixes. */
static const int w = 45;

/* Convenience macro to log a value with a given format. */
#define LV(prefix, format, value) \
    oskar_log_value(log, depth, w, prefix, format, value)

/* Convenience macros to log boolean, integer, string values. */
#define LVB(prefix, value) LV(prefix, "%s", ((value) ? "true" : "false"))
#define LVI(prefix, value) LV(prefix, "%d", value)
#define LVS(prefix, value) LV(prefix, "%s", value)

#define LVS0(key, value) \
    oskar_log_value(log, depth, 0, key, "%s", value)

void oskar_log_settings_simulator(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    oskar_log_message(log, depth, "Simulator settings");
    depth = 1;
    LVB("Double precision", s->sim.double_precision);
    LVB("Keep log file", s->sim.keep_log_file);
    LVI("Num. CUDA devices", s->sim.num_cuda_devices);
    LVI("Max sources per chunk", s->sim.max_sources_per_chunk);
}

static void oskar_log_settings_sky_extended(oskar_Log* log, int depth,
        const oskar_SettingsSkyExtendedSources* s)
{
    if (!(s->FWHM_major == 0.0 && s->FWHM_minor == 0.0))
    {
        LV("FWHM major [arcsec]", "%.3f", s->FWHM_major * 3600*(180/M_PI));
        LV("FWHM minor [arcsec]", "%.3f", s->FWHM_minor * 3600*(180/M_PI));
        LV("Position angle [deg]", "%.3f", s->position_angle * 180/M_PI);
    }
}

static void oskar_log_settings_sky_filter(oskar_Log* log, int depth,
        const oskar_SettingsSkyFilter* f)
{
    if (f->flux_min != 0.0)
        LV("Filter flux min [Jy]", "%.3e", f->flux_min);
    if (f->flux_max != 0.0)
        LV("Filter flux max [Jy]", "%.3e", f->flux_max);
    if (!(f->radius_inner == 0.0 && f->radius_outer >= M_PI / 2.0))
    {
        LV("Filter radius inner [deg]", "%.3f", f->radius_inner * R2D);
        LV("Filter radius outer [deg]", "%.3f", f->radius_outer * R2D);
    }
}

void oskar_log_settings_sky(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0, i = 0;
    oskar_log_message(log, depth, "Sky model settings");

    /* Input OSKAR source file settings. */
    depth = 1;
    if (s->sky.num_sky_files > 0)
    {
        oskar_log_message(log, depth, "Input OSKAR source file(s)");
        ++depth;
        ++depth;
        for (i = 0; i < s->sky.num_sky_files; ++i)
        {
            oskar_log_message(log, depth, "File %2d: %s", i,
                    s->sky.input_sky_file[i]);
        }
        --depth;
        oskar_log_settings_sky_filter(log, depth, &s->sky.input_sky_filter);
        oskar_log_settings_sky_extended(log, depth,
                &s->sky.input_sky_extended_sources);
    }

    /* GSM file settings. */
    depth = 1;
    if (s->sky.gsm_file)
    {
        LVS0("Input GSM file", s->sky.gsm_file);
        ++depth;
        oskar_log_settings_sky_filter(log, depth, &s->sky.gsm_filter);
        oskar_log_settings_sky_extended(log, depth,
                &s->sky.gsm_extended_sources);
    }

    /* Input FITS file settings. */
    depth = 1;
    if (s->sky.num_fits_files > 0)
    {
        oskar_log_message(log, depth, "Input FITS file(s)");
        ++depth;
        ++depth;
        for (i = 0; i < s->sky.num_fits_files; ++i)
        {
            oskar_log_message(log, depth, "File %2d: %s", i,
                    s->sky.fits_file[i]);
        }
        --depth;
        LVI("Downsample factor", s->sky.fits_file_settings.downsample_factor);
        LV("Minimum fraction of peak", "%.2f",
                s->sky.fits_file_settings.min_peak_fraction);
        LV("Noise floor [Jy/pixel]", "%.3e",
                s->sky.fits_file_settings.noise_floor);
        LV("Spectral index", "%.1f", s->sky.fits_file_settings.spectral_index);
    }

    /* Input HEALPix FITS file settings. */
    depth = 1;
    if (s->sky.healpix_fits.num_files > 0)
    {
        oskar_log_message(log, depth, "Input HEALPix FITS file(s)");
        ++depth;
        ++depth;
        for (i = 0; i < s->sky.healpix_fits.num_files; ++i)
        {
            oskar_log_message(log, depth, "File %2d: %s", i,
                    s->sky.healpix_fits.file[i]);
        }
        --depth;
        if (s->sky.healpix_fits.coord_sys == OSKAR_COORD_SYS_GALACTIC)
            LVS("Coordinate system", "Galactic");
        else if (s->sky.healpix_fits.coord_sys == OSKAR_COORD_SYS_EQUATORIAL)
            LVS("Coordinate system", "Equatorial");
        else
            LVS("Coordinate system", "Unknown");
        if (s->sky.healpix_fits.map_units == OSKAR_MAP_UNITS_MK_PER_SR)
            LVS("Map units", "mK/sr");
        else if (s->sky.healpix_fits.map_units == OSKAR_MAP_UNITS_K_PER_SR)
            LVS("Map units", "K/sr");
        else if (s->sky.healpix_fits.map_units == OSKAR_MAP_UNITS_JY)
            LVS("Map units", "Jy/pixel");
        else
            LVS("Map units", "Unknown");
        oskar_log_settings_sky_filter(log, depth, &s->sky.healpix_fits.filter);
        oskar_log_settings_sky_extended(log, depth,
                &s->sky.healpix_fits.extended_sources);
    }

    /* Random power-law generator settings. */
    depth = 1;
    if (s->sky.generator.random_power_law.num_sources != 0)
    {
        const oskar_SettingsSkyGeneratorRandomPowerLaw* gen =
                &s->sky.generator.random_power_law;

        oskar_log_message(log, depth, "Generator (random power law)");
        ++depth;
        LVI("Num. sources", gen->num_sources);
        LV("Flux min [Jy]", "%.3e", gen->flux_min);
        LV("Flux max [Jy]", "%.3e", gen->flux_max);
        LV("Power law index", "%.3f", gen->power);
        LVI("Random seed", gen->seed);
        ++depth;
        oskar_log_settings_sky_filter(log, depth, &gen->filter);
        oskar_log_settings_sky_extended(log, depth, &gen->extended_sources);
    }

    /* Random broken power-law generator settings. */
    depth = 1;
    if (s->sky.generator.random_broken_power_law.num_sources != 0)
    {
        const oskar_SettingsSkyGeneratorRandomBrokenPowerLaw* gen =
                &s->sky.generator.random_broken_power_law;

        oskar_log_message(log, depth, "Generator (random broken power law)");
        ++depth;
        LVI("Num. sources", gen->num_sources);
        LV("Flux min [Jy]", "%.3e", gen->flux_min);
        LV("Flux max [Jy]", "%.3e", gen->flux_max);
        LV("Power law index 1", "%.3f", gen->power1);
        LV("Power law index 2", "%.3f", gen->power2);
        LV("Threshold [Jy]", "%.3e", gen->threshold);
        LVI("Random seed", gen->seed);
        ++depth;
        oskar_log_settings_sky_filter(log, depth, &gen->filter);
        oskar_log_settings_sky_extended(log, depth, &gen->extended_sources);
    }

    /* HEALPix generator settings. */
    depth = 1;
    if (s->sky.generator.healpix.nside != 0)
    {
        const oskar_SettingsSkyGeneratorHealpix* gen =
                &s->sky.generator.healpix;

        oskar_log_message(log, depth, "Generator (HEALPix)");
        ++depth;
        LVI("Nside", gen->nside);
        LVI("(Num. sources)", (12 * (gen->nside) * (gen->nside)));
        ++depth;
        oskar_log_settings_sky_filter(log, depth, &gen->filter);
        oskar_log_settings_sky_extended(log, depth, &gen->extended_sources);
    }

    /* Spectral index override settings. */
    depth = 1;
    if (s->sky.spectral_index.override)
    {
        oskar_log_message(log, depth, "Spectral index overrides");
        ++depth;
        LVB("Override", s->sky.spectral_index.override);
        LV("Reference frequency [Hz]", "%.3e",
                s->sky.spectral_index.ref_frequency_hz);
        LV("Spectral index mean", "%.3f", s->sky.spectral_index.mean);
        LV("Spectral index standard deviation", "%.3f",
                s->sky.spectral_index.std_dev);
        LVI("Random seed", s->sky.spectral_index.seed);
        ++depth;
    }

    /* Output OSKAR sky model file settings. */
    depth = 1;
    LVS0("Output OSKAR sky model text file", s->sky.output_text_file);
    LVS0("Output OSKAR sky model binary file", s->sky.output_binary_file);
}

void oskar_log_settings_observation(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    oskar_log_message(log, depth, "Observation settings");
    depth = 1;
    if (s->obs.num_pointing_levels == 1)
    {
        LV("Phase centre RA [deg]", "%.3f", s->obs.ra0_rad[0] * R2D);
        LV("Phase centre Dec [deg]", "%.3f", s->obs.dec0_rad[0] * R2D);
    }
    else
    {
        int i = 0;
        for (i = 0; i < s->obs.num_pointing_levels; ++i)
        {
            oskar_log_value(log, depth, w, "Phase centre RA [deg]", "(%d) %.3f",
                    i, s->obs.ra0_rad[i] * R2D);
            oskar_log_value(log, depth, w, "Phase centre Dec [deg]", "(%d) %.3f",
                    i, s->obs.dec0_rad[i] * R2D);
        }
    }
    LV("Start frequency [Hz]", "%.3e", s->obs.start_frequency_hz);
    LV("Num. frequency channels", "%d", s->obs.num_channels);
    LV("Frequency inc [Hz]", "%.3e", s->obs.frequency_inc_hz);
    LV("Start time (MJD)", "%f", s->obs.start_mjd_utc);
    LV("Num. time steps", "%d", s->obs.num_time_steps);
    LV("Length [sec]", "%.1f", s->obs.length_seconds);
}


void oskar_log_settings_telescope(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    oskar_log_message(log, depth, "Telescope model settings");
    depth = 1;
    LVS0("Input directory", s->telescope.input_directory);
    LV("Longitude [deg]", "%.1f", s->telescope.longitude_rad * R2D);
    LV("Latitude [deg]", "%.1f", s->telescope.latitude_rad * R2D);
    LV("Altitude [m]", "%.1f", s->telescope.altitude_m);

    /* Aperture array settings. */
    if (s->telescope.station_type == OSKAR_STATION_TYPE_AA)
    {
        const oskar_SettingsApertureArray* aa = &s->telescope.aperture_array;

        LVS("Station type", "Aperture array");
        oskar_log_message(log, depth, "Aperture array settings");
        ++depth;

        /* Array pattern settings. */
        {
            const oskar_SettingsArrayPattern* ap = &aa->array_pattern;
            const oskar_SettingsArrayElement* ae = &ap->element;

            oskar_log_message(log, depth, "Array pattern settings");
            ++depth;
            LVB("Enable array pattern", ap->enable);
            if (ap->enable)
            {
                LVB("Normalise array pattern", ap->normalise);

                /* Array element settings. */
                if (ae->apodisation_type > 0 ||
                        ae->gain > 0.0 ||
                        ae->gain_error_fixed > 0.0 ||
                        ae->gain_error_time > 0.0 ||
                        ae->phase_error_fixed_rad > 0.0 ||
                        ae->phase_error_time_rad > 0.0 ||
                        ae->position_error_xy_m > 0.0 ||
                        ae->x_orientation_error_rad > 0.0 ||
                        ae->y_orientation_error_rad > 0.0)
                {
                    oskar_log_message(log,depth,"Element settings (overrides)");
                    ++depth;
                    switch (ae->apodisation_type)
                    {
                    case 0:
                        LVS("Apodisation type", "None");
                        break;
                    default:
                        LVS("Apodisation type", "Unknown");
                        break;
                    }
                    if (ae->gain > 0.0)
                        LV("Element gain", "%.3f", ae->gain);
                    if (ae->gain_error_fixed > 0.0)
                        LV("Element gain std.dev. (systematic)", "%.3f",
                                ae->gain_error_fixed);
                    if (ae->gain_error_time > 0.0)
                        LV("Element gain std.dev. (time-variable)", "%.3f",
                                ae->gain_error_time);
                    if (ae->phase_error_fixed_rad > 0.0)
                        LV("Element phase std.dev. (systematic) [deg]", "%.3f",
                                ae->phase_error_fixed_rad * R2D);
                    if (ae->phase_error_time_rad > 0.0)
                        LV("Element phase std.dev. (time-variable) [deg]","%.3f",
                                ae->phase_error_time_rad * R2D);
                    if (ae->position_error_xy_m > 0.0)
                        LV("Element (x,y) position std.dev [m]", "%.3f",
                                ae->position_error_xy_m);
                    if (ae->x_orientation_error_rad > 0.0)
                        LV("Element X-dipole orientation std.dev [deg]", "%.3f",
                                ae->x_orientation_error_rad * R2D);
                    if (ae->y_orientation_error_rad > 0.0)
                        LV("Element Y-dipole orientation std.dev [deg]", "%.3f",
                                ae->y_orientation_error_rad * R2D);
                    if (ae->gain > 0.0 || ae->gain_error_fixed > 0.0)
                        LVI("Random seed (systematic gain errors)",
                                ae->seed_gain_errors);
                    if (ae->phase_error_fixed_rad > 0.0)
                        LVI("Random seed (systematic phase errors)",
                                ae->seed_phase_errors);
                    if (ae->gain_error_time > 0.0 ||
                            ae->phase_error_time_rad > 0.0)
                        LVI("Random seed (time-variable errors)",
                                ae->seed_time_variable_errors);
                    if (ae->position_error_xy_m > 0.0)
                        LVI("Random seed (x,y position errors)",
                                ae->seed_position_xy_errors);
                    if (ae->x_orientation_error_rad > 0.0)
                        LVI("Random seed (X-dipole orientation errors)",
                                ae->seed_x_orientation_error);
                    if (ae->y_orientation_error_rad > 0.0)
                        LVI("Random seed (Y-dipole orientation errors)",
                                ae->seed_y_orientation_error);
                    --depth;
                } /* [Element setting overrides] */
            } /* [Enable array pattern] */
            --depth;
        } /* [Array pattern settings] */

        /* Element pattern settings. */
        {
            const oskar_SettingsElementPattern* ep = &aa->element_pattern;

            oskar_log_message(log, depth, "Element pattern settings");
            ++depth;
            LVB("Enable numerical patterns", ep->enable_numerical_patterns);
            if (ep->enable_numerical_patterns)
            {
                const oskar_SettingsElementFit* ef = &ep->fit;

                oskar_log_message(log, depth, "Element pattern fitting "
                        "parameters");
                ++depth;
                LVB("Ignore data at poles", ef->ignore_data_at_pole);
                LVB("Ignore data below horizon", ef->ignore_data_below_horizon);
                LV("Overlap angle [deg]", "%.1f", ef->overlap_angle_rad * R2D);
                LV("Weighting at boundaries", "%.1f", ef->weight_boundaries);
                LV("Weighting in overlap region", "%.1f", ef->weight_overlap);
                oskar_log_message(log, depth, "Common settings (for all surfaces)");
                ++depth;
                LV("Epsilon (single precision)", "%.3e", ef->all.eps_float);
                LV("Epsilon (double precision)", "%.3e", ef->all.eps_double);
                LVB("Search for best fit", ef->all.search_for_best_fit);
                if (ef->all.search_for_best_fit)
                {
                    LV("Average fractional error", "%.4f",
                            ef->all.average_fractional_error);
                    LV("Average fractional error factor increase", "%.1f",
                            ef->all.average_fractional_error_factor_increase);
                }
                else
                {
                    LV("Smoothness factor override", "%.4e",
                            ef->all.smoothness_factor_override);
                }
                --depth;
                --depth;
            } /* [Enable numerical patterns] */

            switch (ep->functional_type)
            {
            case OSKAR_ELEMENT_MODEL_TYPE_ISOTROPIC:
                LVS("Functional pattern type", "Isotropic");
                break;
            case OSKAR_ELEMENT_MODEL_TYPE_GEOMETRIC_DIPOLE:
                LVS("Functional pattern type", "Geometric dipole");
                break;
            default:
                LVS("Functional pattern type", "Unknown");
                break;
            }

            /* Tapering options. */
            {
                oskar_log_message(log, depth, "Tapering options");
                ++depth;
                switch (ep->taper.type)
                {
                case OSKAR_ELEMENT_MODEL_TAPER_NONE:
                    LVS("Tapering type", "None");
                    break;
                case OSKAR_ELEMENT_MODEL_TAPER_COSINE:
                    LVS("Tapering type", "Cosine");
                    ++depth;
                    LV("Cosine power", "%.1f", ep->taper.cosine_power);
                    --depth;
                    break;
                case OSKAR_ELEMENT_MODEL_TAPER_GAUSSIAN:
                    LVS("Tapering type", "Gaussian");
                    ++depth;
                    LV("Gaussian FWHM [deg]", "%.1f",
                            ep->taper.gaussian_fwhm_rad * R2D);
                    --depth;
                    break;
                default:
                    LVS("Tapering type", "Unknown");
                    break;
                }
                --depth;
            } /* [Tapering options] */
            --depth;
        } /* [Element pattern settings] */
        --depth;
    } /* [Aperture array settings] */

    /* Gaussian beam settings. */
    else if (s->telescope.station_type == OSKAR_STATION_TYPE_GAUSSIAN_BEAM)
    {
        LVS("Station type", "Gaussian beam");
        oskar_log_message(log, depth, "Gaussian beam settings");
        ++depth;
        LV("Gaussian FWHM [deg]", "%.4f",
                s->telescope.gaussian_beam.fwhm_deg);
        --depth;
    }

    /* Telescope model output directory. */
    depth = 1;
    LVS0("Output directory", s->telescope.output_directory);
}

void oskar_log_settings_interferometer(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    const oskar_SettingsSystemNoise* n = &s->interferometer.noise;
    oskar_log_message(log, depth, "Interferometer settings");
    depth = 1;

    LV("Channel bandwidth [Hz]", "%.3e", s->interferometer.channel_bandwidth_hz);
    LV("Time average [sec]", "%.2f", s->interferometer.time_average_sec);
    LVI("Num. visibility ave.", s->interferometer.num_vis_ave);
    LVI("Num. fringe ave.", s->interferometer.num_fringe_ave);
    LVB("Use common sky (short baseline approximation)",
            s->interferometer.use_common_sky);

    /* Noise */
    LVB("Enabled system noise", n->enable);
    depth++;
    if (n->enable)
    {
        const char* value;
        LVI("Seed", n->seed);
        /* Noise frequency */
        switch (n->freq.specification)
        {
            case OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL:
            {
                LVS("Frequency specification", "Telescope model");
                break;
            }
            case OSKAR_SYSTEM_NOISE_OBS_SETTINGS:
            {
                LVS("Frequency specification", "Observation settings");
                break;
            }
            case OSKAR_SYSTEM_NOISE_DATA_FILE:
            {
                LVS("Frequency specification", "Data file");
                depth++;
                LVS("Filename", n->freq.file);
                depth--;
                break;
            }
            case OSKAR_SYSTEM_NOISE_RANGE:
            {
                LVS("Frequency specification", "Range");
                depth++;
                LVI("Number", n->freq.number);
                LV("Start", "%.3f", n->freq.start);
                LV("Increment", "%.3f", n->freq.inc);
                depth--;
                break;
            }
            default: { value = "ERROR INVALID VALUE"; break; }
        };

        /* Value */
        switch (n->value.specification)
        {
            case OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL:
            { value = "Telescope model (default priority)"; break; }
            case OSKAR_SYSTEM_NOISE_RMS:
            { value = "RMS flux density"; break; }
            case OSKAR_SYSTEM_NOISE_SENSITIVITY:
            { value = "Sensitivity"; break; }
            case OSKAR_SYSTEM_NOISE_SYS_TEMP:
            {
                value = "System temperature, effective area & system efficiency";
                break;
            }
            default:
            { value = "ERROR INVALID VALUE"; break; }

        };
        LVS("Noise value specification", value);

        /* Print any override setting */
        depth++;
        switch (n->value.specification)
        {
            case OSKAR_SYSTEM_NOISE_RMS:
            {
                const char* override;
                switch (n->value.rms.override)
                {
                    case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
                    { override = "No override"; break;}
                    case OSKAR_SYSTEM_NOISE_DATA_FILE:
                    { override = "Data file"; break;}
                    case OSKAR_SYSTEM_NOISE_RANGE:
                    { override = "Range"; break;}
                    default:
                    { value = "ERROR INVALID VALUE"; break; }
                };
                LVS("Override", override);
                break;
            }
            case OSKAR_SYSTEM_NOISE_SENSITIVITY:
            {
                const char* override;
                switch (n->value.sensitivity.override)
                {
                    case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
                    { override = "No override"; break;}
                    case OSKAR_SYSTEM_NOISE_DATA_FILE:
                    { override = "Data file"; break;}
                    case OSKAR_SYSTEM_NOISE_RANGE:
                    { override = "Range"; break;}
                    default:
                    { value = "ERROR INVALID VALUE"; break; }
                };
                LVS("Override", override);
                break;
            }
            case OSKAR_SYSTEM_NOISE_SYS_TEMP:
            {
                /* TODO */
                break;
            }
            default: { break; }
        };
        depth--;
    }
    depth--;

    LVS0("Output OSKAR visibility file", s->interferometer.oskar_vis_filename);
    LVS0("Output Measurement Set name", s->interferometer.ms_filename);
    LVB("Image simulation output", s->interferometer.image_interferometer_output);
}

void oskar_log_settings_beam_pattern(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    oskar_log_message(log, depth, "Beam pattern settings");
    depth = 1;
    LV("Field-of-view [deg]", "%.3f", s->beam_pattern.fov_deg);
    LVI("Dimension [pixels]", s->beam_pattern.size);
    LVI("Station ID", s->beam_pattern.station_id);
    oskar_log_message(log, depth, "Output OSKAR Image files:");
    ++depth;
    LVS0("Power", s->beam_pattern.oskar_image_power);
    LVS0("Phase", s->beam_pattern.oskar_image_phase);
    LVS0("Complex", s->beam_pattern.oskar_image_complex);
    oskar_log_message(log, --depth, "Output FITS image files:");
    ++depth;
    LVS0("Power", s->beam_pattern.fits_image_power);
    LVS0("Phase", s->beam_pattern.fits_image_phase);
}

void oskar_log_settings_image(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    const char* option;
    oskar_log_message(log, depth, "Image settings");
    depth = 1;
    LV("Field-of-view [deg]", "%.3f", s->image.fov_deg);
    LVI("Dimension (size) [pixels]", s->image.size);
    LVB("Channel snapshots", s->image.channel_snapshots);
    oskar_log_value(log, depth, w, "Channel range", "%i -> %i",
            s->image.channel_range[0], s->image.channel_range[1]);
    LVB("Time snapshots", s->image.time_snapshots);
    oskar_log_value(log, depth, w, "Time range", "%i -> %i",
            s->image.time_range[0], s->image.time_range[1]);

    /* Image type. */
    switch (s->image.image_type)
    {
        case OSKAR_IMAGE_TYPE_POL_LINEAR:
        { option = "Linear (XX,XY,YX,YY)"; break; }
        case OSKAR_IMAGE_TYPE_POL_XX:   { option = "XX"; break; }
        case OSKAR_IMAGE_TYPE_POL_XY:   { option = "XY"; break; }
        case OSKAR_IMAGE_TYPE_POL_YX:   { option = "YX"; break; }
        case OSKAR_IMAGE_TYPE_POL_YY:   { option = "YY"; break; }
        case OSKAR_IMAGE_TYPE_STOKES:
        { option = "Stokes (I,Q,U,V)"; break; }
        case OSKAR_IMAGE_TYPE_STOKES_I: { option = "Stokes I"; break; }
        case OSKAR_IMAGE_TYPE_STOKES_Q: { option = "Stokes Q"; break; }
        case OSKAR_IMAGE_TYPE_STOKES_U: { option = "Stokes U"; break; }
        case OSKAR_IMAGE_TYPE_STOKES_V: { option = "Stokes V"; break; }
        case OSKAR_IMAGE_TYPE_PSF:      { option = "PSF";      break; }
        default:                 { option = "ERROR BAD TYPE"; break; }
    };
    LVS("Image type", option);

    switch (s->image.direction_type)
    {
        case OSKAR_IMAGE_DIRECTION_OBSERVATION:
        {
            LVS("Image centre", "Observation direction (default)");
            break;
        }
        case OSKAR_IMAGE_DIRECTION_RA_DEC:
        {
            LVS("Image centre", "RA, Dec. (override)");
            ++depth;
            LV("Image centre RA", "%.3f", s->image.ra_deg);
            LV("Image centre Dec.", "%.3f", s->image.dec_deg);
            --depth;
            break;
        }
        default:
            break;
    };


    /* Transform type. */
    switch (s->image.transform_type)
    {
        case OSKAR_IMAGE_DFT_2D: { option = "DFT_2D"; break; }
        case OSKAR_IMAGE_DFT_3D: { option = "DFT_3D"; break; }
        case OSKAR_IMAGE_FFT:    { option = "FFT";    break; }
        default:                 { option = "ERROR BAD TYPE"; break; }
    };
    LVS("Transform type", option);

    /* Output files. */
    LVS0("Input OSKAR visibility file", s->image.input_vis_data);
    LVS0("Output OSKAR image file", s->image.oskar_image);
    LVS0("Output FITS image file", s->image.fits_image);
}

#ifdef __cplusplus
}
#endif
