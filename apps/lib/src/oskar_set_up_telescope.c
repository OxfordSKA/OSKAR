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
#include "apps/lib/oskar_telescope_model_config_load.h"
#include "apps/lib/oskar_telescope_model_element_pattern_load.h"
#include "apps/lib/oskar_telescope_model_noise_load.h"

#include "apps/lib/oskar_telescope_model_save.h"

#include "interferometry/oskar_telescope_model_init.h"
#include "interferometry/oskar_telescope_model_analyse.h"
#include "interferometry/oskar_telescope_model_config_override.h"
#include "station/oskar_station_model_load_pointing_file.h"
#include "utility/oskar_get_error_string.h"

#include "utility/oskar_log_error.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_section.h"
#include "utility/oskar_log_value.h"
#include "utility/oskar_log_warning.h"

#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static const int width = 45;

/* Private functions. */
static void oskar_telescope_model_set_metadata(oskar_TelescopeModel *telescope,
        const oskar_Settings* settings);
static void set_station_data(oskar_StationModel* station,
        const oskar_StationModel* parent, int depth,
        const oskar_Settings* settings);
static void save_telescope(oskar_TelescopeModel *telescope,
        const oskar_SettingsTelescope* settings, oskar_Log* log,
        const char* dir, int* status);

void oskar_set_up_telescope(oskar_TelescopeModel *telescope, oskar_Log* log,
        const oskar_Settings* settings, int* status)
{
    int type;
    oskar_log_section(log, "Telescope model");

    /* Initialise the structure in CPU memory. */
    type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_telescope_model_init(telescope, type, OSKAR_LOCATION_CPU, 0, status);

    /* Load the layout and configuration, apply overrides, load noise data. */
    oskar_telescope_model_config_load(telescope, log,
            &settings->telescope, status);
    oskar_telescope_model_config_override(telescope, &settings->telescope,
            status);
    oskar_telescope_model_noise_load(telescope, log, settings, status);

    /* Set telescope model meta-data, including global pointing settings. */
    oskar_telescope_model_set_metadata(telescope, settings);

    /* Apply pointing file override if set. */
    if (settings->obs.pointing_file)
    {
        int i;

        /* Load the same pointing file for every station. */
        oskar_log_message(log, 0, "Loading station pointing file "
                "override '%s'...", settings->obs.pointing_file);
        for (i = 0; i < telescope->num_stations; ++i)
        {
            oskar_station_model_load_pointing_file(&telescope->station[i],
                    settings->obs.pointing_file, status);
        }
    }

    switch (settings->telescope.station_type)
    {
        case OSKAR_STATION_TYPE_AA:
        {
            /* Load element pattern data */
            oskar_telescope_model_element_pattern_load(telescope, log,
                    &settings->telescope, status);

            /* Analyse telescope model to determine whether stations are
             * identical, whether to apply element errors and/or weights. */
            oskar_telescope_model_analyse(telescope, status);
            break;
        }
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            /* FIXME Not correct if FWHM files are used in station model. */
            telescope->identical_stations = OSKAR_TRUE;
            break;
        }
        default:
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            break;
    }
    if (*status) return;

    /* Print summary data. */
    oskar_log_message(log, 0, "Telescope model summary");
    oskar_log_value(log, 1, width, "Num. stations", "%d",
            telescope->num_stations);
    oskar_log_value(log, 1, width, "Max station size", "%d",
            telescope->max_station_size);
    oskar_log_value(log, 1, width, "Max station depth", "%d",
            telescope->max_station_depth);
    oskar_log_value(log, 1, width, "Identical stations", "%s",
            telescope->identical_stations ? "true" : "false");

    /* Save the telescope configuration in a new directory, if required. */
    save_telescope(telescope, &settings->telescope, log,
            settings->telescope.output_directory, status);
}


static void oskar_telescope_model_set_metadata(oskar_TelescopeModel *telescope,
        const oskar_Settings* settings)
{
    int i, seed;
    const oskar_SettingsApertureArray* aa = &settings->telescope.aperture_array;
    telescope->ra0_rad        = settings->obs.ra0_rad[0];
    telescope->dec0_rad       = settings->obs.dec0_rad[0];
    telescope->use_common_sky = settings->interferometer.use_common_sky;
    telescope->bandwidth_hz   = settings->interferometer.channel_bandwidth_hz;
    telescope->time_average_sec = settings->interferometer.time_average_sec;
    telescope->wavelength_metres = 0.0; /* This is set on a per-channel basis. */
    seed = aa->array_pattern.element.seed_time_variable_errors;
    telescope->seed_time_variable_station_element_errors = seed;
    for (i = 0; i < telescope->num_stations; ++i)
    {
        /* Set station data (recursively, for all child stations too). */
        set_station_data(&telescope->station[i], NULL, 0, settings);
    }
}

static void set_station_data(oskar_StationModel* station,
        const oskar_StationModel* parent, int depth,
        const oskar_Settings* settings)
{
    int i = 0;
    const oskar_SettingsApertureArray* aa = &settings->telescope.aperture_array;
    station->station_type = settings->telescope.station_type;
    if (parent)
    {
        station->longitude_rad = parent->longitude_rad;
        station->latitude_rad = parent->latitude_rad;
        station->altitude_m = parent->altitude_m;
    }
    station->enable_array_pattern = aa->array_pattern.enable;
    station->normalise_beam = aa->array_pattern.normalise;
    station->gaussian_beam_fwhm_deg =
            settings->telescope.gaussian_beam.fwhm_deg;
    station->use_polarised_elements =
            !(aa->element_pattern.functional_type ==
                    OSKAR_ELEMENT_MODEL_TYPE_ISOTROPIC);

    /* Set element pattern data, if element structure exists. */
    if (station->element_pattern)
    {
        station->element_pattern->type = aa->element_pattern.functional_type;
        station->element_pattern->taper_type = aa->element_pattern.taper.type;
        station->element_pattern->cos_power =
                aa->element_pattern.taper.cosine_power;
        station->element_pattern->gaussian_fwhm_rad =
                aa->element_pattern.taper.gaussian_fwhm_rad;
    }

    /* Set pointing data based on station depth in hierarchy. */
    i = (depth < settings->obs.num_pointing_levels) ? depth :
            settings->obs.num_pointing_levels - 1;
    station->beam_longitude_rad = settings->obs.ra0_rad[i];
    station->beam_latitude_rad = settings->obs.dec0_rad[i];

    /* Recursively set data for child stations. */
    if (station->child)
    {
        for (i = 0; i < station->num_elements; ++i)
        {
            set_station_data(&station->child[i], station, depth + 1, settings);
        }
    }
}

static void save_telescope(oskar_TelescopeModel *telescope,
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
    oskar_telescope_model_save(telescope, dir, status);
    if (*status)
    {
        oskar_log_error(log, "Failed to save telescope "
                "configuration (%s).", oskar_get_error_string(*status));
    }
}

#ifdef __cplusplus
}
#endif
