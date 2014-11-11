/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include "apps/lib/oskar_set_up_telescope.h"
#include "apps/lib/oskar_telescope_load.h"

#include "apps/lib/oskar_telescope_save.h"

#include <oskar_telescope.h>
#include <oskar_get_error_string.h>
#include <oskar_log.h>
#include <oskar_cmath.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Private functions. */
static void oskar_telescope_set_metadata(oskar_Telescope *telescope,
        const oskar_Settings* settings, int* status);
static void set_station_data(oskar_Station* station,
        const oskar_Station* parent, int depth,
        const oskar_Settings* settings);

static void save_telescope(oskar_Telescope *telescope,
        const oskar_SettingsTelescope* settings, oskar_Log* log,
        const char* dir, int* status);

void oskar_telescope_log_summary(const oskar_Telescope* telescope,
        oskar_Log* log, int* status);

void oskar_station_log_summary(const oskar_Station* station,
        oskar_Log* log, int depth, int* status);

oskar_Telescope* oskar_set_up_telescope(const oskar_Settings* settings,
        oskar_Log* log, int* status)
{
    int type;
    oskar_Telescope* telescope;

    /* Check all inputs. */
    if (!settings || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check if safe to proceed. */
    if (*status) return 0;

    oskar_log_section(log, 'M', "Telescope model");

    /* Initialise the structure in CPU memory. */
    type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    telescope = oskar_telescope_create(type, OSKAR_CPU, 0, status);

    /* Load the telescope model and then apply overrides. */
    oskar_telescope_load(telescope, log, settings, status);
    oskar_telescope_config_override(telescope, &settings->telescope, status);

    /* Set telescope model meta-data, including global pointing settings. */
    oskar_telescope_set_metadata(telescope, settings, status);

    /* Apply pointing file override if set. */
    if (settings->obs.pointing_file)
    {
        oskar_log_message(log, 'M', 0, "Loading station pointing file "
                "override '%s'...", settings->obs.pointing_file);
        oskar_telescope_load_pointing_file(telescope,
                settings->obs.pointing_file, status);
    }

    /* Convert all phase centre ICRS to CIRS coordinates at start time. */
    /*oskar_telescope_convert_phase_centre_coords(telescope,
            OSKAR_SPHERICAL_TYPE_EQUATORIAL_ICRS,
            OSKAR_SPHERICAL_TYPE_EQUATORIAL_CIRS,
            settings->obs.start_mjd_utc,
            settings->obs.delta_tai_utc_sec,
            settings->obs.delta_ut1_utc_sec, status);
     */

    /* Analyse telescope model to determine whether stations are
     * identical, whether to apply element errors and/or weights. */
    oskar_telescope_analyse(telescope, status);
    if (*status) return telescope;

    /* Set error flag if number of stations is less than 2. */
    if (oskar_telescope_num_stations(telescope) < 2)
    {
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE;
        oskar_log_error(log, "Insufficient number of stations to form an "
                "interferometer baseline (%d stations found).",
                oskar_telescope_num_stations(telescope));
    }

    /* Print summary data. */
    oskar_telescope_log_summary(telescope, log, status);

    /* Save the telescope configuration in a new directory, if required. */
    save_telescope(telescope, &settings->telescope, log,
            settings->telescope.output_directory, status);

    return telescope;
}


void oskar_telescope_log_summary(const oskar_Telescope* telescope,
        oskar_Log* log, int* status)
{
    int num_stations = 0;

    /* Check all inputs. */
    if (!telescope || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Print top-level data. */
    num_stations = oskar_telescope_num_stations(telescope);
    oskar_log_message(log, 'M', 0, "Telescope model summary");
    oskar_log_value(log, 'M', 1, "Num. stations", "%d", num_stations);
    oskar_log_value(log, 'M', 1, "Max station size", "%d",
            oskar_telescope_max_station_size(telescope));
    oskar_log_value(log, 'M', 1, "Max station depth", "%d",
            oskar_telescope_max_station_depth(telescope));
    oskar_log_value(log, 'M', 1, "Identical stations", "%s",
            oskar_telescope_identical_stations(telescope) ? "true" : "false");

#if 0
    /* Switch on whether stations are identical. */
    if (oskar_telescope_identical_stations(telescope) && num_stations > 0)
    {
        /* Print station summary for first station. */
        oskar_log_message(log, 1, "Station model summary");
        oskar_station_log_summary(oskar_telescope_station_const(telescope, 0),
                log, 1, status);
    }
    else
    {
        int i = 0;

        /* Loop over top-level stations to print summary for each. */
        for (i = 0; i < num_stations; ++i)
        {
            oskar_log_message(log, 1, "Station %d model summary", i);
            oskar_station_log_summary(oskar_telescope_station_const(telescope,
                    i), log, 1, status);
        }
    }
#endif
}

void oskar_station_log_summary(const oskar_Station* station,
        oskar_Log* log, int depth, int* status)
{
    int d1 = 0;

    /* Check all inputs. */
    if (!station || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get next level down. */
    d1 = depth + 1;

    /* Print data for this level. */
    if (oskar_station_type(station) == OSKAR_STATION_TYPE_AA)
    {
        int i = 0, num_elements = 0;

        /* Print station type and number of elements. */
        num_elements = oskar_station_num_elements(station);
        oskar_log_value(log, 'M', d1, "Num. elements", "%d", num_elements);

        /* Print child station data. */
        if (oskar_station_has_child(station))
        {
            oskar_log_value(log, 'M', d1, "Identical child stations", "%s",
                    oskar_station_identical_children(station) ?
                            "true" : "false");

            /* Switch on whether child stations are identical. */
            if (oskar_station_identical_children(station) && num_elements > 0)
            {
                /* Print station summary for first child station. */
                oskar_log_message(log, 'M', d1, "Station model summary");
                oskar_station_log_summary(oskar_station_child_const(station, 0),
                        log, d1, status);
            }
            else
            {
                /* Loop over child stations to print summary for each. */
                for (i = 0; i < num_elements; ++i)
                {
                    oskar_log_message(log, 'M', d1, "Station %d model summary", i);
                    oskar_station_log_summary(
                            oskar_station_child_const(station, i), log, d1,
                            status);
                }
            }
        }
    }
    else if (oskar_station_type(station) == OSKAR_STATION_TYPE_ISOTROPIC)
    {
        oskar_log_value(log, 'M', d1, "Station type", "Isotropic");
    }
    else if (oskar_station_type(station) == OSKAR_STATION_TYPE_VLA_PBCOR)
    {
        oskar_log_value(log, 'M', d1, "Station type", "VLA (PBCOR)");
    }
    else if (oskar_station_type(station) == OSKAR_STATION_TYPE_GAUSSIAN_BEAM)
    {
        oskar_log_value(log, 'M', d1, "Station type", "Gaussian beam");
    }
}

static void oskar_telescope_set_metadata(oskar_Telescope *telescope,
        const oskar_Settings* settings, int* status)
{
    int i, num_stations;
    const oskar_SettingsApertureArray* aa = &settings->telescope.aperture_array;

    if (*status) return;

    oskar_telescope_set_phase_centre(telescope,
            OSKAR_SPHERICAL_TYPE_EQUATORIAL,
            settings->obs.phase_centre_lon_rad[0],
            settings->obs.phase_centre_lat_rad[0]);
    oskar_telescope_set_polar_motion(telescope,
            (settings->obs.pm_x_arcsec / 3600.0) * M_PI / 180.0,
            (settings->obs.pm_y_arcsec / 3600.0) * M_PI / 180.0);
    oskar_telescope_set_allow_station_beam_duplication(telescope,
            settings->telescope.allow_station_beam_duplication);
    oskar_telescope_set_smearing_values(telescope,
            settings->interferometer.channel_bandwidth_hz,
            settings->interferometer.time_average_sec);
    oskar_telescope_set_uv_filter(telescope,
            settings->interferometer.uv_filter_min,
            settings->interferometer.uv_filter_max,
            settings->interferometer.uv_filter_units);
    oskar_telescope_set_random_seed(telescope,
            aa->array_pattern.element.seed_time_variable_errors);
    num_stations = oskar_telescope_num_stations(telescope);
    for (i = 0; i < num_stations; ++i)
    {
        oskar_Station* station;

        /* Set station data (recursively, for all child stations too). */
        station = oskar_telescope_station(telescope, i);
        set_station_data(station, NULL, 0, settings);
    }
}

static void set_station_data(oskar_Station* station,
        const oskar_Station* parent, int depth,
        const oskar_Settings* settings)
{
    int i = 0;
    const oskar_SettingsApertureArray* aa = &settings->telescope.aperture_array;
    oskar_station_set_station_type(station, settings->telescope.station_type);
    oskar_station_set_normalise_final_beam(station,
            settings->telescope.normalise_beams_at_phase_centre);
    if (parent)
    {
        oskar_station_set_position(station,
                oskar_station_lon_rad(parent),
                oskar_station_lat_rad(parent),
                oskar_station_alt_metres(parent));
    }
    oskar_station_set_enable_array_pattern(station, aa->array_pattern.enable);
    oskar_station_set_normalise_array_pattern(station,
            aa->array_pattern.normalise);
    oskar_station_set_gaussian_beam(station,
            settings->telescope.gaussian_beam.fwhm_deg * M_PI / 180.0,
            settings->telescope.gaussian_beam.ref_freq_hz);

    /* Set element pattern data for all element types (if they exist). */
    for (i = 0; i < oskar_station_num_element_types(station); ++i)
    {
        const oskar_SettingsElementPattern* ep;
        oskar_Element* element;
        ep = &aa->element_pattern;
        element = oskar_station_element(station, i);
        oskar_element_set_element_type(element, ep->functional_type);
        oskar_element_set_dipole_length(element, ep->dipole_length,
                ep->dipole_length_units);
        oskar_element_set_taper_type(element, ep->taper.type);
        oskar_element_set_cosine_power(element, ep->taper.cosine_power);
        oskar_element_set_gaussian_fwhm_rad(element, ep->taper.gaussian_fwhm_rad);
    }

    /* Set pointing data based on station depth in hierarchy. */
    i = (depth < settings->obs.num_pointing_levels) ? depth :
            settings->obs.num_pointing_levels - 1;
    oskar_station_set_phase_centre(station,
            OSKAR_SPHERICAL_TYPE_EQUATORIAL_ICRS,
            settings->obs.phase_centre_lon_rad[i],
            settings->obs.phase_centre_lat_rad[i]);

    /* Recursively set data for child stations. */
    if (oskar_station_has_child(station))
    {
        int num_elements;
        num_elements = oskar_station_num_elements(station);
        for (i = 0; i < num_elements; ++i)
        {
            set_station_data(oskar_station_child(station, i), station,
                    depth + 1, settings);
        }
    }
}

static void save_telescope(oskar_Telescope *telescope,
        const oskar_SettingsTelescope* settings, oskar_Log* log,
        const char* dir, int* status)
{
    /* Check if safe to proceed. */
    if (*status) return;

    /* Do nothing if no or empty directory string specified. */
    if (!dir || !strlen(dir)) return;

    /* Check that the input and output directories are different. */
    if (!strcmp(dir, settings->input_directory))
    {
        oskar_log_warning(log, "Will not overwrite input telescope directory!");
        return;
    }

    oskar_log_message(log, 'M', 1, "Writing telescope model to disk as: %s", dir);
    oskar_telescope_save(telescope, dir, status);
    if (*status)
    {
        oskar_log_error(log, "Failed to save telescope "
                "configuration (%s).", oskar_get_error_string(*status));
    }
}

#ifdef __cplusplus
}
#endif
