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
        if (!(s->sky.input_sky_filter.radius_inner == 0.0 &&
                s->sky.input_sky_filter.radius_outer >= M_PI/2.0))
        {
            LV("Filter radius inner [deg]", "%.3f",
                    s->sky.input_sky_filter.radius_inner * R2D);
            LV("Filter radius outer [deg]", "%.3f",
                    s->sky.input_sky_filter.radius_outer * R2D);
        }
        if (s->sky.input_sky_filter.flux_min != 0.0)
            LV("Filter flux min [Jy]", "%.3e",
                    s->sky.input_sky_filter.flux_min);
        if (s->sky.input_sky_filter.flux_max != 0.0)
            LV("Filter flux max [Jy]", "%.3e",
                    s->sky.input_sky_filter.flux_max);
    }

    /* GSM file settings. */
    depth = 1;
    LVS0("Input GSM file", s->sky.gsm_file);
    ++depth;
    if (!(s->sky.gsm_filter.radius_inner == 0.0 &&
            s->sky.gsm_filter.radius_outer >= M_PI/2.0))
    {
        LV("Filter radius inner [deg]", "%.3f",
                s->sky.gsm_filter.radius_inner * R2D);
        LV("Filter radius outer [deg]", "%.3f",
                s->sky.gsm_filter.radius_outer * R2D);
    }
    if (s->sky.gsm_filter.flux_min != 0.0)
        LV("Filter flux min [Jy]", "%.3e", s->sky.gsm_filter.flux_min);
    if (s->sky.gsm_filter.flux_max != 0.0)
        LV("Filter flux max [Jy]", "%.3e", s->sky.gsm_filter.flux_max);

    /* Output OSKAR source file settings. */
    depth = 1;
    LVS0("Output OSKAR source file", s->sky.output_sky_file);

    /* Random power-law generator settings. */
    depth = 1;
    if (s->sky.generator.random_power_law.num_sources != 0)
    {
        oskar_log_message(log, depth, "Generator (random power law)");
        ++depth;
        LVI("Num. sources", s->sky.generator.random_power_law.num_sources);
        LV("Flux min [Jy]", "%.3e", s->sky.generator.random_power_law.flux_min);
        LV("Flux max [Jy]", "%.3e", s->sky.generator.random_power_law.flux_max);
        LV("Power law index", "%.3f", s->sky.generator.random_power_law.power);
        LVI("Random seed", s->sky.generator.random_power_law.seed);
        ++depth;
        if (!(s->sky.generator.random_power_law.filter.radius_inner == 0.0 &&
                s->sky.generator.random_power_law.filter.radius_outer >= M_PI/2.0))
        {
            LV("Filter radius inner [deg]", "%.3f",
                    s->sky.generator.random_power_law.filter.radius_inner * R2D);
            LV("Filter radius outer [deg]", "%.3f",
                    s->sky.generator.random_power_law.filter.radius_outer * R2D);
        }
        if (s->sky.generator.random_power_law.filter.flux_min != 0.0)
            LV("Filter flux min [Jy]", "%.3e",
                    s->sky.generator.random_power_law.filter.flux_min);
        if (s->sky.generator.random_power_law.filter.flux_max != 0.0)
            LV("Filter flux max [Jy]", "%.3e",
                    s->sky.generator.random_power_law.filter.flux_max);
    }

    /* Random broken power-law generator settings. */
    depth = 1;
    if (s->sky.generator.random_broken_power_law.num_sources != 0)
    {
        oskar_log_message(log, depth, "Generator (random broken power law)");
        ++depth;
        LVI("Num. sources", s->sky.generator.random_broken_power_law.num_sources);
        LV("Flux min [Jy]", "%.3e", s->sky.generator.random_broken_power_law.flux_min);
        LV("Flux max [Jy]", "%.3e", s->sky.generator.random_broken_power_law.flux_max);
        LV("Power law index 1", "%.3f", s->sky.generator.random_broken_power_law.power1);
        LV("Power law index 2", "%.3f", s->sky.generator.random_broken_power_law.power2);
        LV("Threshold [Jy]", "%.3f", s->sky.generator.random_broken_power_law.threshold);
        LVI("Random seed", s->sky.generator.random_broken_power_law.seed);
        ++depth;
        if (!(s->sky.generator.random_broken_power_law.filter.radius_inner == 0.0 &&
                s->sky.generator.random_broken_power_law.filter.radius_outer >= M_PI/2.0))
        {
            LV("Filter radius inner [deg]", "%.3f",
                    s->sky.generator.random_broken_power_law.filter.radius_inner * R2D);
            LV("Filter radius outer [deg]", "%.3f",
                    s->sky.generator.random_broken_power_law.filter.radius_outer * R2D);
        }
        if (s->sky.generator.random_broken_power_law.filter.flux_min != 0.0)
            LV("Filter flux min [Jy]", "%.3e",
                    s->sky.generator.random_broken_power_law.filter.flux_min);
        if (s->sky.generator.random_broken_power_law.filter.flux_max != 0.0)
            LV("Filter flux max [Jy]", "%.3e",
                    s->sky.generator.random_broken_power_law.filter.flux_max);
    }

    /* HEALPix generator settings. */
    depth = 1;
    if (s->sky.generator.healpix.nside != 0)
    {
        int n;
        n = 12 * (int)pow(s->sky.generator.healpix.nside, 2.0);
        oskar_log_message(log, depth, "Generator (HEALPix)");
        ++depth;
        LVI("Nside", s->sky.generator.healpix.nside);
        LVI("(Num. sources)", n);
        ++depth;
        if (!(s->sky.generator.healpix.filter.radius_inner == 0.0 &&
                s->sky.generator.healpix.filter.radius_outer >= M_PI/2.0))
        {
            LV("Filter radius inner [deg]", "%.3f",
                    s->sky.generator.healpix.filter.radius_inner * R2D);
            LV("Filter radius outer [deg]", "%.3f",
                    s->sky.generator.healpix.filter.radius_outer * R2D);
        }
        if (s->sky.generator.healpix.filter.flux_min != 0.0)
            LV("Filter flux min [Jy]", "%.3e",
                    s->sky.generator.healpix.filter.flux_min);
        if (s->sky.generator.healpix.filter.flux_max != 0.0)
            LV("Filter flux max [Jy]", "%.3e",
                    s->sky.generator.healpix.filter.flux_max);
    }
}

void oskar_log_settings_observation(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    oskar_log_message(log, depth, "Observation settings");
    depth = 1;
    LV("Phase centre RA [deg]", "%.3f", s->obs.ra0_rad * R2D);
    LV("Phase centre Dec [deg]", "%.3f", s->obs.dec0_rad * R2D);
    LV("Start frequency [Hz]", "%.3e", s->obs.start_frequency_hz);
    LV("Num. frequency channels", "%d", s->obs.num_channels);
    LV("Frequency inc [Hz]", "%.3e", s->obs.frequency_inc_hz);
    LV("Start time (MJD)", "%f", s->obs.start_mjd_utc);
    LV("Num. time steps", "%d", s->obs.num_time_steps);
    LV("Length [sec]", "%f", s->obs.length_seconds);
}


void oskar_log_settings_telescope(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    oskar_log_message(log, depth, "Telescope model settings");
    depth = 1;
    LVS0("Telescope directory", s->telescope.config_directory);
    LV("Longitude [deg]", "%.1f", s->telescope.longitude_rad * R2D);
    LV("Latitude [deg]", "%.1f", s->telescope.latitude_rad * R2D);
    LV("Altitude [m]", "%.1f", s->telescope.altitude_m);
    LVB("Use common sky", s->telescope.use_common_sky);

    /* Station model settings. */
    oskar_log_message(log, depth, "Station model settings");
    ++depth;
    LVS("Station type", s->telescope.station.station_type ==
            OSKAR_STATION_TYPE_DISH ? "Dishes" : "Aperture Arrays");
    LVB("Use polarised elements", s->telescope.station.use_polarised_elements);
    LVB("Ignore custom element patterns",
            s->telescope.station.ignore_custom_element_patterns);
    LVB("Evaluate array factor",
            s->telescope.station.evaluate_array_factor);
    LVB("Evaluate element factor",
            s->telescope.station.evaluate_element_factor);
    LVB("Normalise array beam", s->telescope.station.normalise_beam);

    /* Element model settings. */
    if (s->telescope.station.element.gain > 0.0 ||
            s->telescope.station.element.gain_error_fixed > 0.0 ||
            s->telescope.station.element.gain_error_time > 0.0 ||
            s->telescope.station.element.phase_error_fixed_rad > 0.0 ||
            s->telescope.station.element.phase_error_time_rad > 0.0 ||
            s->telescope.station.element.position_error_xy_m > 0.0 ||
            s->telescope.station.element.x_orientation_error_rad > 0.0 ||
            s->telescope.station.element.y_orientation_error_rad > 0.0)
    {
        oskar_log_message(log, depth, "Element settings (overrides)");
    }
    ++depth;
    if (s->telescope.station.element.gain > 0.0)
        LV("Element gain", "%.3f", s->telescope.station.element.gain);
    if (s->telescope.station.element.gain_error_fixed > 0.0)
        LV("Element gain std.dev. (systematic)", "%.3f",
                s->telescope.station.element.gain_error_fixed);
    if (s->telescope.station.element.gain_error_time > 0.0)
        LV("Element gain std.dev. (time-variable)", "%.3f",
                s->telescope.station.element.gain_error_time);
    if (s->telescope.station.element.phase_error_fixed_rad > 0.0)
        LV("Element phase std.dev. (systematic) [deg]", "%.3f",
                s->telescope.station.element.phase_error_fixed_rad * R2D);
    if (s->telescope.station.element.phase_error_time_rad > 0.0)
        LV("Element phase std.dev. (time-variable) [deg]", "%.3f",
                s->telescope.station.element.phase_error_time_rad * R2D);
    if (s->telescope.station.element.position_error_xy_m > 0.0)
        LV("Element (x,y) position std.dev [m]", "%.3f",
                s->telescope.station.element.position_error_xy_m);
    if (s->telescope.station.element.x_orientation_error_rad > 0.0)
        LV("Element X-dipole orientation std.dev [deg]", "%.3f",
                s->telescope.station.element.x_orientation_error_rad * R2D);
    if (s->telescope.station.element.y_orientation_error_rad > 0.0)
        LV("Element Y-dipole orientation std.dev [deg]", "%.3f",
                s->telescope.station.element.y_orientation_error_rad * R2D);
    if (s->telescope.station.element.gain > 0.0 ||
            s->telescope.station.element.gain_error_fixed > 0.0)
        LVI("Random seed (systematic gain errors)",
                s->telescope.station.element.seed_gain_errors);
    if (s->telescope.station.element.phase_error_fixed_rad > 0.0)
        LVI("Random seed (systematic phase errors)",
                s->telescope.station.element.seed_phase_errors);
    if (s->telescope.station.element.gain_error_time > 0.0 ||
            s->telescope.station.element.phase_error_time_rad > 0.0)
        LVI("Random seed (time-variable errors)",
                s->telescope.station.element.seed_time_variable_errors);
    if (s->telescope.station.element.position_error_xy_m > 0.0)
        LVI("Random seed (x,y position errors)",
                s->telescope.station.element.seed_position_xy_errors);
    if (s->telescope.station.element.x_orientation_error_rad > 0.0)
        LVI("Random seed (X-dipole orientation errors)",
                s->telescope.station.element.seed_x_orientation_error);
    if (s->telescope.station.element.y_orientation_error_rad > 0.0)
        LVI("Random seed (Y-dipole orientation errors)",
                s->telescope.station.element.seed_y_orientation_error);

    depth = 1;
    LVS0("Output telescope directory", s->telescope.output_config_directory);
}

void oskar_log_settings_interferometer(oskar_Log* log, const oskar_Settings* s)
{
    int depth = 0;
    const oskar_SettingsSystemNoise* n = &s->interferometer.noise;
    oskar_log_message(log, depth, "Interferometer settings");
    depth = 1;

    LV("Channel bandwidth [Hz]", "%f", s->interferometer.channel_bandwidth_hz);
    LVI("Num. visibility ave.", s->interferometer.num_vis_ave);
    LVI("Num. fringe ave.", s->interferometer.num_fringe_ave);

    /* Noise */
    LVB("Enabled system noise", n->enable);
    depth++;
    if (n->enable)
    {
        const char* value;
        LVI("Seed", n->seed);
        LVB("Apply area projection", n->area_projection);
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
            { value = "RMS Flux density"; break; }
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
                // TODO
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

void oskar_log_settings(oskar_Log* log, const oskar_Settings* s,
        const char* filename)
{
    int depth = 0;

    /* Print name of settings file. */
    LVS0("OSKAR settings file", filename);

    /* Print simulator settings. */
    oskar_log_settings_simulator(log, s);

    /* Print sky settings. */
    oskar_log_settings_sky(log, s);

    /* Print observation settings. */
    oskar_log_settings_observation(log, s);

    /* Print telescope settings. */
    oskar_log_settings_telescope(log, s);

    /* Print interferometer settings */
    if (s->interferometer.oskar_vis_filename || s->interferometer.ms_filename)
        oskar_log_settings_interferometer(log, s);

    /* Print beam pattern settings. */
    if (s->beam_pattern.size > 0)
        oskar_log_settings_beam_pattern(log, s);

    /* Print image settings. */
    if (s->image.size > 0)
        oskar_log_settings_image(log, s);
}

#ifdef __cplusplus
}
#endif
