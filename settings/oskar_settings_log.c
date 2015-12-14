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

#include <oskar_settings_log.h>

#include <private_log.h>
#include <oskar_log.h>
#include <oskar_telescope.h>
#include <oskar_image.h>
#include <oskar_Settings.h>

#include <stdio.h>
#include <oskar_cmath.h>
#include <string.h>
#include <float.h>

#define R2D 180.0/M_PI

#ifdef __cplusplus
extern "C" {
#endif

/* Convenience macro to log a value with a given format. */
#define LV(prefix, format, value) oskar_log_value(log, 'M', depth, prefix, format, value)

/* Convenience macros to log boolean, integer, string values. */
#define LVB(prefix, value) LV(prefix, "%s", ((value) ? "true" : "false"))
#define LVI(prefix, value) LV(prefix, "%d", value)
#define LVS(prefix, value) LV(prefix, "%s", value)

/* Width 0 value list message */
#define LVS0(key, value) oskar_log_message(log, 'M', depth, "%s: %s", key, value)

void oskar_log_settings_simulator(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    oskar_log_message(log, 'M', depth, "Simulator settings");
    depth = 1;
    LVB("Double precision", s->sim.double_precision);
    LVB("Keep log file", s->sim.keep_log_file);
    LVB("Log progress status", s->sim.write_status_to_log_file);
    LVI("Num. CUDA devices", s->sim.num_cuda_devices);
    LVI("Max sources per chunk", s->sim.max_sources_per_chunk);
}

static void oskar_log_settings_sky_polarisation(oskar_Log* log, int depth,
        const oskar_SettingsSkyPolarisation* s)
{
    if (s->mean_pol_fraction != 0.0 || s->std_pol_fraction != 0.0)
    {
        LV("Mean polarisation fraction", "%.3f", s->mean_pol_fraction);
        LV("Std.dev. polarisation fraction", "%.3f", s->std_pol_fraction);
        LV("Mean polarisation angle [deg]", "%.3f", s->mean_pol_angle_rad * 180/M_PI);
        LV("Std.dev. polarisation angle [deg]", "%.3 f", s->std_pol_angle_rad * 180/M_PI);
        LVI("Random seed", s->seed);
    }
}

static void oskar_log_settings_sky_extended(oskar_Log* log, int depth,
        const oskar_SettingsSkyExtendedSources* s)
{
    if (!(s->FWHM_major_rad == 0.0 && s->FWHM_minor_rad == 0.0))
    {
        LV("FWHM major [arcsec]", "%.3f", s->FWHM_major_rad * 3600*(180/M_PI));
        LV("FWHM minor [arcsec]", "%.3f", s->FWHM_minor_rad * 3600*(180/M_PI));
        LV("Position angle [deg]", "%.3f", s->position_angle_rad * 180/M_PI);
    }
}

static void oskar_log_settings_sky_filter(oskar_Log* log, int depth,
        const oskar_SettingsSkyFilter* f)
{
    if (f->flux_min != 0.0)
        LV("Filter flux min [Jy]", "%.3e", f->flux_min);
    if (f->flux_max != 0.0)
        LV("Filter flux max [Jy]", "%.3e", f->flux_max);
    if (!(f->radius_inner_rad == 0.0 && f->radius_outer_rad >= M_PI / 2.0))
    {
        LV("Filter radius inner [deg]", "%.3f", f->radius_inner_rad * R2D);
        LV("Filter radius outer [deg]", "%.3f", f->radius_outer_rad * R2D);
    }
}

void oskar_log_settings_sky(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0, i = 0;
    oskar_log_message(log, 'M', depth, "Sky model settings");

    /* Input OSKAR sky model file settings. */
    depth = 1;
    if (s->sky.oskar_sky_model.num_files > 0)
    {
        oskar_log_message(log, 'M', depth, "Input OSKAR sky model file(s)");
        ++depth;
        ++depth;
        for (i = 0; i < s->sky.oskar_sky_model.num_files; ++i)
        {
            oskar_log_message(log, 'M', depth, "File %2d: %s", i,
                    s->sky.oskar_sky_model.file[i]);
        }
        --depth;
        oskar_log_settings_sky_filter(log, depth,
                &s->sky.oskar_sky_model.filter);
        oskar_log_settings_sky_extended(log, depth,
                &s->sky.oskar_sky_model.extended_sources);
    }

    /* GSM file settings. */
    depth = 1;
    if (s->sky.gsm.file)
    {
        LVS0("Input GSM file", s->sky.gsm.file);
        ++depth;
        oskar_log_settings_sky_filter(log, depth, &s->sky.gsm.filter);
        oskar_log_settings_sky_extended(log, depth,
                &s->sky.gsm.extended_sources);
    }

    /* Input FITS file settings. */
    depth = 1;
    if (s->sky.fits_image.num_files > 0)
    {
        oskar_log_message(log, 'M', depth, "Input FITS file(s)");
        ++depth;
        ++depth;
        for (i = 0; i < s->sky.fits_image.num_files; ++i)
        {
            oskar_log_message(log, 'M', depth, "File %2d: %s", i,
                    s->sky.fits_image.file[i]);
        }
        --depth;
        LVI("Downsample factor", s->sky.fits_image.downsample_factor);
        LV("Minimum fraction of peak", "%.2f",
                s->sky.fits_image.min_peak_fraction);
        LV("Noise floor [Jy/pixel]", "%.3e",
                s->sky.fits_image.noise_floor);
        LV("Spectral index", "%.1f", s->sky.fits_image.spectral_index);
    }

    /* Input HEALPix FITS file settings. */
    depth = 1;
    if (s->sky.healpix_fits.num_files > 0)
    {
        oskar_log_message(log, 'M', depth, "Input HEALPix FITS file(s)");
        ++depth;
        ++depth;
        for (i = 0; i < s->sky.healpix_fits.num_files; ++i)
        {
            oskar_log_message(log, 'M', depth, "File %2d: %s", i,
                    s->sky.healpix_fits.file[i]);
        }
        --depth;
        if (s->sky.healpix_fits.coord_sys == OSKAR_SPHERICAL_TYPE_GALACTIC)
            LVS("Coordinate system", "Galactic");
        else if (s->sky.healpix_fits.coord_sys == OSKAR_SPHERICAL_TYPE_EQUATORIAL)
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

        oskar_log_message(log, 'M', depth, "Generator (random power law)");
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

        oskar_log_message(log, 'M', depth, "Generator (random broken power law)");
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

    /* Grid generator settings. */
    depth = 1;
    if (s->sky.generator.grid.side_length != 0)
    {
        const oskar_SettingsSkyGeneratorGrid* gen = &s->sky.generator.grid;

        oskar_log_message(log, 'M', depth, "Generator (grid at phase centre)");
        ++depth;
        LVI("Side length", gen->side_length);
        LV("Field-of-view [deg]", "%.3f", gen->fov_rad * R2D);
        LV("Mean Stokes I flux [Jy]", "%.3e", gen->mean_flux_jy);
        LV("Std.dev. Stokes I flux [Jy]", "%.3e", gen->std_flux_jy);
        LVI("Random seed", gen->seed);
        ++depth;
        oskar_log_settings_sky_polarisation(log, depth, &gen->pol);
        oskar_log_settings_sky_extended(log, depth, &gen->extended_sources);
    }

    /* HEALPix generator settings. */
    depth = 1;
    if (s->sky.generator.healpix.nside != 0)
    {
        const oskar_SettingsSkyGeneratorHealpix* gen =
                &s->sky.generator.healpix;

        oskar_log_message(log, 'M', depth, "Generator (HEALPix)");
        ++depth;
        LVI("Nside", gen->nside);
        LVI("(Num. sources)", (12 * (gen->nside) * (gen->nside)));
        LV("Amplitude [Jy]", "%.3e", gen->amplitude);
        ++depth;
        oskar_log_settings_sky_filter(log, depth, &gen->filter);
        oskar_log_settings_sky_extended(log, depth, &gen->extended_sources);
    }

    /* Spectral index override settings. */
    depth = 1;
    if (s->sky.spectral_index.override)
    {
        oskar_log_message(log, 'M', depth, "Spectral index overrides");
        ++depth;
        LVB("Override", s->sky.spectral_index.override);
        LV("Reference frequency [Hz]", "%.3e",
                s->sky.spectral_index.ref_frequency_hz);
        LV("Spectral index mean", "%.3f", s->sky.spectral_index.mean);
        LV("Spectral index standard deviation", "%.3f",
                s->sky.spectral_index.std_dev);
        LVI("Random seed", s->sky.spectral_index.seed);
    }

    /* Common source flux filtering settings. */
    depth = 1;
    if (s->sky.common_flux_filter_min_jy != 0.0 ||
            s->sky.common_flux_filter_max_jy != FLT_MAX)
    {
        oskar_log_message(log, 'M', depth, "Common source flux filtering settings");
        ++depth;
        LV("Filter flux min [Jy]", "%.3e", s->sky.common_flux_filter_min_jy);
        LV("Filter flux max [Jy]", "%.3e", s->sky.common_flux_filter_max_jy);
    }

    /* Output OSKAR sky model file settings. */
    depth = 1;
    if (s->sky.output_text_file)
    {
        LVS0("Output OSKAR sky model text file", s->sky.output_text_file);
    }
    if (s->sky.output_binary_file)
    {
        LVS0("Output OSKAR sky model binary file", s->sky.output_binary_file);
    }

    /* Advanced settings. */
    if (s->sky.zero_failed_gaussians)
    {
        LVB("Remove failed Gaussian sources", s->sky.zero_failed_gaussians);
    }
}

void oskar_log_settings_observation(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    oskar_log_message(log, 'M', depth, "Observation settings");
    depth = 1;
    if (s->obs.num_pointing_levels == 1)
    {
        LV("Phase centre RA [deg]", "%.3f", s->obs.phase_centre_lon_rad[0] * R2D);
        LV("Phase centre Dec [deg]", "%.3f", s->obs.phase_centre_lat_rad[0] * R2D);
    }
    else
    {
        int i = 0;
        for (i = 0; i < s->obs.num_pointing_levels; ++i)
        {
            oskar_log_value(log, 'M', depth, "Phase centre RA [deg]", "(%d) %.3f",
                    i, s->obs.phase_centre_lon_rad[i] * R2D);
            oskar_log_value(log, 'M', depth, "Phase centre Dec [deg]", "(%d) %.3f",
                    i, s->obs.phase_centre_lat_rad[i] * R2D);
        }
    }
    if (s->obs.pointing_file)
    {
        LVS0("Station pointing file", s->obs.pointing_file);
    }
    LV("Start frequency [Hz]", "%.3e", s->obs.start_frequency_hz);
    LV("Num. frequency channels", "%d", s->obs.num_channels);
    LV("Frequency inc [Hz]", "%.3e", s->obs.frequency_inc_hz);
    LV("Start time (MJD)", "%f", s->obs.start_mjd_utc);
    LV("Num. time steps", "%d", s->obs.num_time_steps);
    LV("Length [sec]", "%.1f", s->obs.length_sec);
}


void oskar_log_settings_telescope(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    oskar_log_message(log, 'M', depth, "Telescope model settings");
    depth = 1;
    LVS0("Input directory", s->telescope.input_directory);
    LV("Longitude [deg]", "%.1f", s->telescope.longitude_rad * R2D);
    LV("Latitude [deg]", "%.1f", s->telescope.latitude_rad * R2D);
    LV("Altitude [m]", "%.1f", s->telescope.altitude_m);
    LVB("Normalise beams at phase centre",
            s->telescope.normalise_beams_at_phase_centre);
    LVS("Polarisation mode",
            s->telescope.pol_mode == OSKAR_POL_MODE_FULL ? "Full" : "Scalar");
    LVB("Allow station beam duplication",
            s->telescope.allow_station_beam_duplication);

    /* Aperture array settings. */
    if (s->telescope.station_type == OSKAR_STATION_TYPE_AA)
    {
        const oskar_SettingsApertureArray* aa = &s->telescope.aperture_array;

        LVS("Station type", "Aperture array");
        oskar_log_message(log, 'M', depth, "Aperture array settings");
        ++depth;

        /* Array pattern settings. */
        {
            const oskar_SettingsArrayPattern* ap = &aa->array_pattern;
            const oskar_SettingsArrayElement* ae = &ap->element;

            oskar_log_message(log, 'M', depth, "Array pattern settings");
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
                    oskar_log_message(log, 'M', depth,"Element settings (overrides)");
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
                        LV("Element gain", "%.3e", ae->gain);
                    if (ae->gain_error_fixed > 0.0)
                        LV("Element gain std.dev. (systematic)", "%.3e",
                                ae->gain_error_fixed);
                    if (ae->gain_error_time > 0.0)
                        LV("Element gain std.dev. (time-variable)", "%.3e",
                                ae->gain_error_time);
                    if (ae->phase_error_fixed_rad > 0.0)
                        LV("Element phase std.dev. (systematic) [deg]", "%.3e",
                                ae->phase_error_fixed_rad * R2D);
                    if (ae->phase_error_time_rad > 0.0)
                        LV("Element phase std.dev. (time-variable) [deg]","%.3e",
                                ae->phase_error_time_rad * R2D);
                    if (ae->position_error_xy_m > 0.0)
                        LV("Element (x,y) position std.dev [m]", "%.3e",
                                ae->position_error_xy_m);
                    if (ae->x_orientation_error_rad > 0.0)
                        LV("Element X-dipole orientation std.dev [deg]", "%.3e",
                                ae->x_orientation_error_rad * R2D);
                    if (ae->y_orientation_error_rad > 0.0)
                        LV("Element Y-dipole orientation std.dev [deg]", "%.3e",
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

            oskar_log_message(log, 'M', depth, "Element pattern settings");
            ++depth;
            LVB("Enable numerical patterns", ep->enable_numerical_patterns);

            switch (ep->functional_type)
            {
            case OSKAR_ELEMENT_TYPE_ISOTROPIC:
                LVS("Functional pattern type", "Isotropic");
                break;
            case OSKAR_ELEMENT_TYPE_GEOMETRIC_DIPOLE:
                LVS("Functional pattern type", "Geometric dipole");
                break;
            case OSKAR_ELEMENT_TYPE_DIPOLE:
            {
                LVS("Functional pattern type", "Dipole");
                LV("Dipole length", "%.3f", ep->dipole_length);
                switch (ep->dipole_length_units)
                {
                case OSKAR_WAVELENGTHS:
                    LVS("Dipole length units", "Wavelengths");
                    break;
                case OSKAR_METRES:
                    LVS("Dipole length units", "Metres");
                    break;
                }
                break;
            }
            default:
                LVS("Functional pattern type", "Unknown");
                break;
            }

            /* Tapering options. */
            {
                oskar_log_message(log, 'M', depth, "Tapering options");
                ++depth;
                switch (ep->taper.type)
                {
                case OSKAR_ELEMENT_TAPER_NONE:
                    LVS("Tapering type", "None");
                    break;
                case OSKAR_ELEMENT_TAPER_COSINE:
                    LVS("Tapering type", "Cosine");
                    ++depth;
                    LV("Cosine power", "%.1f", ep->taper.cosine_power);
                    --depth;
                    break;
                case OSKAR_ELEMENT_TAPER_GAUSSIAN:
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
        oskar_log_message(log, 'M', depth, "Gaussian beam settings");
        ++depth;
        LV("Gaussian FWHM [deg]", "%.4f",
                s->telescope.gaussian_beam.fwhm_deg);
        LV("Reference frequency [Hz]", "%.3e",
                s->telescope.gaussian_beam.ref_freq_hz);
        --depth;
    }
    else if (s->telescope.station_type == OSKAR_STATION_TYPE_ISOTROPIC)
    {
        LVS("Station type", "Isotropic (STATION BEAM DISABLED!)");
    }

    /* Telescope model output directory. */
    depth = 1;
    if (s->telescope.output_directory)
    {
        LVS0("Output directory", s->telescope.output_directory);
    }
}

void oskar_log_settings_interferometer(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    const oskar_SettingsSystemNoise* n = &s->interferometer.noise;
    oskar_log_message(log, 'M', depth, "Interferometer settings");
    depth = 1;

    LV("Channel bandwidth [Hz]", "%.3e", s->interferometer.channel_bandwidth_hz);
    LV("Time average [sec]", "%.2f", s->interferometer.time_average_sec);
    LVI("Max. time samples per block",
            s->interferometer.max_time_samples_per_block);
    if (s->interferometer.uv_filter_min > 0.0)
        LV("UV range filter min", "%.3f", s->interferometer.uv_filter_min);
    if (s->interferometer.uv_filter_max >= 0.0)
        LV("UV range filter max", "%.3f", s->interferometer.uv_filter_max);
    if (s->interferometer.uv_filter_min > 0.0 ||
            s->interferometer.uv_filter_max >= 0.0)
        LVS("UV range filter units", s->interferometer.uv_filter_units ==
                OSKAR_METRES ? "Metres" : "Wavelengths");

    /* Noise */
    LVB("System noise enabled", n->enable);
    depth++;
    if (n->enable)
    {
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
            default: { break; }
        };

        switch (n->rms.specification)
        {
        case OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL:
        {
            LVS("RMS specification", "Telescope model");
            break;
        }
        case OSKAR_SYSTEM_NOISE_OBS_SETTINGS:
        {
            LVS("RMS specification", "Observation settings");
            break;
        }
        case OSKAR_SYSTEM_NOISE_DATA_FILE:
        {
            LVS("RMS specification", "Data file");
            depth++;
            LVS("Filename", n->freq.file);
            depth--;
            break;
        }
        case OSKAR_SYSTEM_NOISE_RANGE:
        {
            LVS("RMS specification", "Range");
            depth++;
            LV("Start", "%.3f", n->rms.end);
            LV("End", "%.3f", n->rms.end);
            depth--;
            break;
        }
        default: { break; }
        }
    }
    depth--;

    LVS0("Output OSKAR visibility file", s->interferometer.oskar_vis_filename);
    LVS0("Output Measurement Set name", s->interferometer.ms_filename);
    if (s->interferometer.force_polarised_ms)
        LVB("Force polarised MS", s->interferometer.force_polarised_ms);
}

void oskar_log_settings_beam_pattern(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    const oskar_SettingsBeamPattern* b = &s->beam_pattern;
    oskar_log_message(log, 'M', depth, "Beam pattern settings");
    depth = 1;
    LVB("All stations", b->all_stations);
    if (!b->all_stations)
        LVI("Number of stations to evaluate", b->num_active_stations);
    LVS("Coordinate frame type",
            b->coord_frame_type == OSKAR_BEAM_PATTERN_FRAME_HORIZON ?
                    "Horizon" : "Equatorial");
    switch (b->coord_grid_type)
    {
    case OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE:
    {
        LVS("Coordinate (grid) type", "Beam image");
        oskar_log_value(log, 'M', ++depth, "Image dimensions [pixels]",
                "%i, %i", b->size[0], b->size[1]);
        if (b->coord_frame_type == OSKAR_BEAM_PATTERN_FRAME_EQUATORIAL)
            oskar_log_value(log, 'M', depth, "Field-of-view [deg]",
                    "%.3f, %.3f", b->fov_deg[0], b->fov_deg[1]);
        break;
    }
    case OSKAR_BEAM_PATTERN_COORDS_SKY_MODEL:
    {
        LVS("Coordinate (grid) type", "Sky model");
        ++depth;
        LVS("Input file", b->sky_model);
        break;
    }
    case OSKAR_BEAM_PATTERN_COORDS_HEALPIX:
    {
        LVS("Coordinate (grid) type", "HEALPix");
        ++depth;
        LVI("Nside", b->nside);
        break;
    }
    };

    depth = 1;
    LVS("Output root path name", b->root_path);
    oskar_log_message(log, 'M', depth++, "Output options:");
    LVB("Separate time and channel", b->separate_time_and_channel);
    LVB("Average time and channel", b->average_time_and_channel);
    switch (b->average_single_axis)
    {
    case OSKAR_BEAM_PATTERN_AVERAGE_CHANNEL:
        LVS("Average single axis", "Channel");
        break;
    case OSKAR_BEAM_PATTERN_AVERAGE_TIME:
        LVS("Average single axis", "Time");
        break;
    default:
        LVS("Average single axis", "None");
        break;
    }
    depth = 1;
    oskar_log_message(log, 'M', depth++, "Per-station outputs:");
    oskar_log_message(log, 'M', depth++, "Text file:");
    if (b->separate_time_and_channel)
    {
        LVB("Raw (complex) pattern", b->station_text_raw_complex);
        LVB("Amplitude pattern", b->station_text_amp);
        LVB("Phase pattern", b->station_text_phase);
    }
    LVB("Auto-correlation Stokes I power pattern",
            b->station_text_auto_power_stokes_i);
    depth--;
    if (b->coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE)
    {
        oskar_log_message(log, 'M', depth++, "FITS image:");
        if (b->separate_time_and_channel)
        {
            LVB("Amplitude pattern", b->station_fits_amp);
            LVB("Phase pattern", b->station_fits_phase);
        }
        LVB("Auto-correlation Stokes I power pattern",
                b->station_fits_auto_power_stokes_i);
    }
    depth = 1;
    oskar_log_message(log, 'M', depth++, "Telescope outputs:");
    oskar_log_message(log, 'M', depth++, "Text file:");
    LVB("Cross-correlation Stokes I raw power pattern",
            b->telescope_text_cross_power_stokes_i_raw_complex);
    LVB("Cross-correlation Stokes I amplitude power pattern",
            b->telescope_text_cross_power_stokes_i_amp);
    LVB("Cross-correlation Stokes I phase power pattern",
            b->telescope_text_cross_power_stokes_i_phase);
    depth--;
    if (b->coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE)
    {
        oskar_log_message(log, 'M', depth++, "FITS image:");
        LVB("Cross-correlation Stokes I amplitude power pattern",
                b->telescope_fits_cross_power_stokes_i_amp);
        LVB("Cross-correlation Stokes I phase power pattern",
                b->telescope_fits_cross_power_stokes_i_phase);
    }
}

void oskar_log_settings_image(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    const char* option;
    oskar_log_message(log, 'M', depth, "Image settings");
    depth = 1;
    LV("Field-of-view [deg]", "%.3f", s->image.fov_deg);
    LVI("Dimension (size) [pixels]", s->image.size);
    LVB("Channel snapshots", s->image.channel_snapshots);
    oskar_log_value(log, 'M', depth, "Channel range", "%i -> %i",
            s->image.channel_range[0], s->image.channel_range[1]);
    LVB("Time snapshots", s->image.time_snapshots);
    oskar_log_value(log, 'M', depth, "Time range", "%i -> %i",
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
    LVS0("Output FITS image file", s->image.fits_image);
}

void oskar_log_settings_ionosphere(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    oskar_log_message(log, 'M', depth, "Ionosphere (Z Jones) settings");
    depth = 1;
    LVB("Enabled", s->ionosphere.enable);
    LV("Minimum elevation (deg)", "%.3f%", s->ionosphere.min_elevation * 180./M_PI);
    LV("TEC0", "%.3f%", s->ionosphere.TEC0);
    LVI("Number of TID screens", s->ionosphere.num_TID_screens);

    if (s->ionosphere.TECImage.fits_file)
    {
        oskar_log_message(log, 'M', depth, "TEC image settings");
        depth++;
        LVI("Station index", s->ionosphere.TECImage.stationID);
        LVB("Beam centred", s->ionosphere.TECImage.beam_centred);
        LVI("Image size", s->ionosphere.TECImage.size);
        LV("FOV (deg)", "%.3f", s->ionosphere.TECImage.fov_rad*(180./M_PI));
        if (s->ionosphere.TECImage.fits_file)
            LVS("FITS image", s->ionosphere.TECImage.fits_file);
    }
}

void oskar_log_settings_element_fit(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    const oskar_SettingsElementFit* ef = &s->element_fit;

    oskar_log_message(log, 'M', depth, "Element pattern fitting settings");
    ++depth;
    LVS("Input CST file", ef->input_cst_file);
    LVS("Input scalar file", ef->input_scalar_file);
    LVS("Output FITS image file", ef->fits_image);
    LV("Frequency [Hz]", "%.5e", ef->frequency_hz);
    LVS("Polarisation type", ef->pol_type == 1 ? "X" :
            ef->pol_type == 2 ? "Y" : "XY");
    LVB("Ignore data at poles", ef->ignore_data_at_pole);
    LVB("Ignore data below horizon", ef->ignore_data_below_horizon);
    LV("Average fractional error", "%.4f",
            ef->average_fractional_error);
    LV("Average fractional error factor increase", "%.1f",
            ef->average_fractional_error_factor_increase);
    LVS("Output telescope/station directory", ef->output_directory);
}

#ifdef __cplusplus
}
#endif
