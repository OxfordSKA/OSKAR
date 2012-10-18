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

#include "apps/lib/oskar_set_up_telescope.h"
#include "apps/lib/oskar_telescope_model_config_load.h"
#include "apps/lib/oskar_telescope_model_element_pattern_load.h"
#include "apps/lib/oskar_telescope_model_noise_load.h"

#include "apps/lib/oskar_telescope_model_save.h"

#include "interferometry/oskar_telescope_model_init.h"
#include "interferometry/oskar_telescope_model_analyse.h"
#include "interferometry/oskar_telescope_model_config_override.h"
#include "utility/oskar_get_error_string.h"

#include "utility/oskar_log_error.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_section.h"
#include "utility/oskar_log_value.h"
#include "utility/oskar_log_warning.h"

#include "utility/oskar_Mem.h"
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

/* private functions */
static void set_metadata(oskar_TelescopeModel *telescope,
        const oskar_Settings* settings);
static int save_telescope(oskar_TelescopeModel *telescope,
        const oskar_SettingsTelescope* settings, oskar_Log* log, const char* dir);

int oskar_set_up_telescope(oskar_TelescopeModel *telescope, oskar_Log* log,
        const oskar_Settings* settings)
{
    int err = 0, type;
    oskar_log_section(log, "Telescope model");

    /* Initialise the structure in CPU memory. */
    type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_telescope_model_init(telescope, type, OSKAR_LOCATION_CPU, 0, &err);
    if (err) return err;

    /* Load the layout and configuration */
    err = oskar_telescope_model_config_load(telescope, log, &settings->telescope);
    if (err) return err;

    /* Apply configuration overrides */
    oskar_telescope_model_config_override(telescope, &settings->telescope, &err);
    if (err) return err;

    /* Load noise data */
    err = oskar_telescope_model_noise_load(telescope, log, settings);
    if (err) return err;

    switch (settings->telescope.station_type)
    {
        case OSKAR_STATION_TYPE_AA:
        {
            /* Load element pattern data */
            err = oskar_telescope_model_element_pattern_load(telescope, log,
                    &settings->telescope);
            if (err) return err;

            /* Analyse telescope model to determine whether stations are identical,
             * whether to apply element errors and/or weights. */
            oskar_telescope_model_analyse(telescope, &err);
            if (err) return err;
            break;
        }
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            /* FIXME If FWHM files are used in the station model, this won't be correct. */
            telescope->identical_stations = OSKAR_TRUE;
            break;
        }
        default:
            return OSKAR_ERR_SETTINGS_TELESCOPE;
    }

    /* Set telescope model meta-data */
    set_metadata(telescope, settings);

    /* Print summary data. */
    oskar_log_message(log, 0, "Telescope model summary");
    oskar_log_value(log, 1, width, "Num. stations", "%d", telescope->num_stations);
    oskar_log_value(log, 1, width, "Max station size", "%d", telescope->max_station_size);
    oskar_log_value(log, 1, width, "Identical stations", "%s", telescope->identical_stations ? "true" : "false");

    /* Save the telescope configuration in a new directory, if required. */
    err = save_telescope(telescope, &settings->telescope, log,
            settings->telescope.output_directory);
    if (err) return err;

    return OSKAR_SUCCESS;
}


static void set_metadata(oskar_TelescopeModel *telescope, const oskar_Settings* settings)
{
    int i, seed;
    const oskar_SettingsApertureArray* aa = &settings->telescope.aperture_array;
    telescope->ra0_rad        = settings->obs.ra0_rad;
    telescope->dec0_rad       = settings->obs.dec0_rad;
    telescope->use_common_sky = settings->interferometer.use_common_sky;
    telescope->bandwidth_hz   = settings->interferometer.channel_bandwidth_hz;
    telescope->time_average_sec = settings->interferometer.time_average_sec;
    telescope->wavelength_metres = 0.0; /* This is set on a per-channel basis. */
    seed = aa->array_pattern.element.seed_time_variable_errors;
    telescope->seed_time_variable_station_element_errors = seed;
    for (i = 0; i < telescope->num_stations; ++i)
    {
        telescope->station[i].station_type = settings->telescope.station_type;
        telescope->station[i].ra0_rad = telescope->ra0_rad;
        telescope->station[i].dec0_rad = telescope->dec0_rad;
        telescope->station[i].enable_array_pattern =
                aa->array_pattern.enable;
        telescope->station[i].normalise_beam =
                settings->telescope.aperture_array.array_pattern.normalise;
        telescope->station[i].gaussian_beam_fwhm_deg =
                settings->telescope.gaussian_beam.fwhm_deg;
        telescope->station[i].use_polarised_elements =
                !(aa->element_pattern.functional_type ==
                        OSKAR_ELEMENT_MODEL_TYPE_ISOTROPIC);

        /* Set element pattern data, if element structure exists. */
        if (telescope->station[i].element_pattern)
        {
            telescope->station[i].element_pattern->type =
                    aa->element_pattern.functional_type;
            telescope->station[i].element_pattern->taper_type =
                    aa->element_pattern.taper.type;
            telescope->station[i].element_pattern->cos_power =
                    aa->element_pattern.taper.cosine_power;
            telescope->station[i].element_pattern->gaussian_fwhm_rad =
                    aa->element_pattern.taper.gaussian_fwhm_rad;
        }
    }
}

static int save_telescope(oskar_TelescopeModel *telescope,
        const oskar_SettingsTelescope* settings, oskar_Log* log, const char* dir)
{
    int err = OSKAR_SUCCESS;

    /* No output directory specified = Do nothing. */
    if (!dir) return OSKAR_SUCCESS;

    /* Empty output directory specified = Do nothing. */
    if (!strlen(dir)) return OSKAR_SUCCESS;

    /* Check that the input and output directories are different. */
    if (!strcmp(dir, settings->input_directory))
    {
        oskar_log_warning(log, "Will not overwrite input telescope directory!");
        return OSKAR_SUCCESS;
    }

    oskar_log_message(log, 1, "Writing telescope model to disk as: %s", dir);
    err = oskar_telescope_model_save(telescope, dir);
    if (err)
    {
        oskar_log_error(log, "Failed to save telescope "
                "configuration (%s).", oskar_get_error_string(err));
        return err;
    }

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
