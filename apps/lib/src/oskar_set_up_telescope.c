/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

static const int width = 45;

/* Private functions. */
static void oskar_telescope_set_metadata(oskar_Telescope *telescope,
        const oskar_Settings* settings, int* status);
static void set_station_data(oskar_Station* station,
        const oskar_Station* parent, int depth,
        const oskar_Settings* settings);
static void save_telescope(oskar_Telescope *telescope,
        const oskar_SettingsTelescope* settings, oskar_Log* log,
        const char* dir, int* status);

oskar_Telescope* oskar_set_up_telescope(oskar_Log* log,
        const oskar_Settings* settings, int* status)
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

    oskar_log_section(log, "Telescope model");

    /* Initialise the structure in CPU memory. */
    type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    telescope = oskar_telescope_create(type, OSKAR_LOCATION_CPU, 0, status);

    /* Load the layout and configuration, apply overrides, load noise data. */
    oskar_telescope_load(telescope, log, settings, status);
    oskar_telescope_config_override(telescope, &settings->telescope, status);

    /* Set telescope model meta-data, including global pointing settings. */
    oskar_telescope_set_metadata(telescope, settings, status);

    /* Apply pointing file override if set. */
    if (settings->obs.pointing_file)
    {
        oskar_log_message(log, 0, "Loading station pointing file "
                "override '%s'...", settings->obs.pointing_file);
        oskar_telescope_load_pointing_file(telescope,
                settings->obs.pointing_file, status);
    }

    /* Analyse telescope model to determine whether stations are
     * identical, whether to apply element errors and/or weights. */
    oskar_telescope_analyse(telescope, status);
    if (*status) return telescope;

    /* Print summary data. */
    oskar_log_message(log, 0, "Telescope model summary");
    oskar_log_value(log, 1, width, "Num. stations", "%d",
            oskar_telescope_num_stations(telescope));
    oskar_log_value(log, 1, width, "Max station size", "%d",
            oskar_telescope_max_station_size(telescope));
    oskar_log_value(log, 1, width, "Max station depth", "%d",
            oskar_telescope_max_station_depth(telescope));
    oskar_log_value(log, 1, width, "Identical stations", "%s",
            oskar_telescope_identical_stations(telescope) ?
                    "true" : "false");

    /* Save the telescope configuration in a new directory, if required. */
    save_telescope(telescope, &settings->telescope, log,
            settings->telescope.output_directory, status);

    return telescope;
}


static void oskar_telescope_set_metadata(oskar_Telescope *telescope,
        const oskar_Settings* settings, int* status)
{
    int i, num_stations;
    const oskar_SettingsApertureArray* aa = &settings->telescope.aperture_array;

    if (*status) return;

    oskar_telescope_set_phase_centre(telescope,
            settings->obs.ra0_rad[0], settings->obs.dec0_rad[0]);
    oskar_telescope_set_common_horizon(telescope,
            settings->interferometer.use_common_sky);
    oskar_telescope_set_smearing_values(telescope,
            settings->interferometer.channel_bandwidth_hz,
            settings->interferometer.time_average_sec);
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
    if (parent)
    {
        oskar_station_set_position(station,
                oskar_station_longitude_rad(parent),
                oskar_station_latitude_rad(parent),
                oskar_station_altitude_m(parent));
    }
    oskar_station_set_enable_array_pattern(station, aa->array_pattern.enable);
    oskar_station_set_normalise_beam(station, aa->array_pattern.normalise);
    oskar_station_set_gaussian_beam_fwhm_rad(station,
            settings->telescope.gaussian_beam.fwhm_deg * M_PI / 180.0);
    oskar_station_set_use_polarised_elements(station,
            !(aa->element_pattern.functional_type ==
                    OSKAR_ELEMENT_TYPE_ISOTROPIC));

    /* Set element pattern data, if element structure exists. */
    if (oskar_station_has_element(station))
    {
        const oskar_SettingsElementPattern* ep;
        oskar_Element* element;
        ep = &aa->element_pattern;
        element = oskar_station_element(station, 0);
        oskar_element_set_element_type(element, ep->functional_type);
        oskar_element_set_taper_type(element, ep->taper.type);
        oskar_element_set_cos_power(element, ep->taper.cosine_power);
        oskar_element_set_gaussian_fwhm_rad(element, ep->taper.gaussian_fwhm_rad);
    }

    /* Set pointing data based on station depth in hierarchy. */
    i = (depth < settings->obs.num_pointing_levels) ? depth :
            settings->obs.num_pointing_levels - 1;
    oskar_station_set_phase_centre(station, OSKAR_SPHERICAL_TYPE_EQUATORIAL,
            settings->obs.ra0_rad[i], settings->obs.dec0_rad[i]);

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
    /* No output directory specified = Do nothing. */
    if (!dir) return;

    /* Empty output directory specified = Do nothing. */
    if (!strlen(dir)) return;

    /* Check that the input and output directories are different. */
    if (!strcmp(dir, settings->input_directory))
    {
        oskar_log_warning(log, "Will not overwrite input telescope directory!");
        return;
    }

    oskar_log_message(log, 1, "Writing telescope model to disk as: %s", dir);
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
