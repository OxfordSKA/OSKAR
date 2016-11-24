/*
 * Copyright (c) 2011-2016, The University of Oxford
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

#include "oskar_set_up_telescope.h"

#include "telescope/oskar_telescope.h"
#include "utility/oskar_get_error_string.h"
#include "log/oskar_log.h"
#include "math/oskar_cmath.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Private functions. */
static void set_station_data(oskar_Station* station,
        const oskar_Settings_old* s);

oskar_Telescope* oskar_set_up_telescope(const oskar_Settings_old* settings,
        oskar_Log* log, int* status)
{
    int i, num_stations, noise_freq_spec, noise_rms_spec, type;
    oskar_Telescope* t;
    const oskar_SettingsArrayElement* element;
    if (*status) return 0;

    /* Create an empty telescope model. */
    type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    t = oskar_telescope_create(type, OSKAR_CPU, 0, status);

    /* Set telescope model options from settings. */
    /* This must be done before station data are loaded. */
    oskar_telescope_set_allow_station_beam_duplication(t,
            settings->telescope.allow_station_beam_duplication);
    oskar_telescope_set_channel_bandwidth(t,
            settings->interferometer.channel_bandwidth_hz);
    oskar_telescope_set_time_average(t,
            settings->interferometer.time_average_sec);
    oskar_telescope_set_uv_filter(t,
            settings->interferometer.uv_filter_min,
            settings->interferometer.uv_filter_max,
            settings->interferometer.uv_filter_units);
    oskar_telescope_set_pol_mode(t, settings->telescope.pol_mode);
    oskar_telescope_set_enable_numerical_patterns(t,
            settings->telescope.aperture_array.element_pattern.
            enable_numerical_patterns);
    oskar_telescope_set_enable_noise(t,
            settings->interferometer.noise.enable,
            settings->interferometer.noise.seed);

    /************************************************************************/
    /* Load telescope model folders to define the stations. */
    oskar_telescope_load(t, settings->telescope.input_directory, log, status);
    if (*status) return t;

    /* Set error flag if no stations were found. */
    num_stations = oskar_telescope_num_stations(t);
    if (num_stations < 1)
    {
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE;
        oskar_log_error(log, "Telescope model is empty.");
        return t;
    }

    /************************************************************************/
    /* Set remaining options from settings. */
    /* These must be done after the stations have been defined. */
    oskar_telescope_set_phase_centre(t,
            OSKAR_SPHERICAL_TYPE_EQUATORIAL,
            settings->obs.phase_centre_lon_rad[0],
            settings->obs.phase_centre_lat_rad[0]);
    oskar_telescope_set_station_type(t, settings->telescope.station_type);
    oskar_telescope_set_gaussian_station_beam_width(t,
            settings->telescope.gaussian_beam.fwhm_deg,
            settings->telescope.gaussian_beam.ref_freq_hz);

    /* Apply system noise overrides from settings, if required. */
    noise_freq_spec = settings->interferometer.noise.freq.specification;
    noise_rms_spec = settings->interferometer.noise.rms.specification;
    switch (noise_freq_spec)
    {
    case OSKAR_SYSTEM_NOISE_RANGE:
        oskar_telescope_set_noise_freq(t,
                settings->interferometer.noise.freq.start,
                settings->interferometer.noise.freq.inc,
                settings->interferometer.noise.freq.number,
                status);
        break;
    case OSKAR_SYSTEM_NOISE_OBS_SETTINGS:
        oskar_telescope_set_noise_freq(t,
                settings->obs.start_frequency_hz,
                settings->obs.frequency_inc_hz,
                settings->obs.num_channels,
                status);
        break;
    case OSKAR_SYSTEM_NOISE_DATA_FILE:
        oskar_telescope_set_noise_freq_file(t,
                settings->interferometer.noise.freq.file, status);
        break;
    }
    switch (noise_rms_spec)
    {
    case OSKAR_SYSTEM_NOISE_RANGE:
        oskar_telescope_set_noise_rms(t,
                settings->interferometer.noise.rms.start,
                settings->interferometer.noise.rms.end, status);
        break;
    case OSKAR_SYSTEM_NOISE_DATA_FILE:
        oskar_telescope_set_noise_rms_file(t,
                settings->interferometer.noise.rms.file, status);
        break;
    }
    for (i = 0; i < num_stations; ++i)
        set_station_data(oskar_telescope_station(t, i), settings);

    /* Apply element level overrides. */
    element = &settings->telescope.aperture_array.array_pattern.element;

    /* Override station element systematic/fixed gain errors if required. */
    if (element->gain > 0.0 || element->gain_error_fixed > 0.0)
    {
        for (i = 0; i < num_stations; ++i)
            oskar_station_override_element_gains(
                    oskar_telescope_station(t, i),
                    element->seed_gain_errors, element->gain,
                    element->gain_error_fixed, status);
    }

    /* Override station element time-variable gain errors if required. */
    if (element->gain_error_time > 0.0)
    {
        for (i = 0; i < num_stations; ++i)
            oskar_station_override_element_time_variable_gains(
                    oskar_telescope_station(t, i),
                    element->gain_error_time, status);
    }

    /* Override station element systematic/fixed phase errors if required. */
    if (element->phase_error_fixed_rad > 0.0)
    {
        for (i = 0; i < num_stations; ++i)
            oskar_station_override_element_phases(
                    oskar_telescope_station(t, i),
                    element->seed_phase_errors,
                    element->phase_error_fixed_rad, status);
    }

    /* Override station element time-variable phase errors if required. */
    if (element->phase_error_time_rad > 0.0)
    {
        for (i = 0; i < num_stations; ++i)
            oskar_station_override_element_time_variable_phases(
                    oskar_telescope_station(t, i),
                    element->phase_error_time_rad, status);
    }

    /* Override station element position errors if required. */
    if (element->position_error_xy_m > 0.0)
    {
        for (i = 0; i < num_stations; ++i)
            oskar_station_override_element_xy_position_errors(
                    oskar_telescope_station(t, i),
                    element->seed_position_xy_errors,
                    element->position_error_xy_m, status);
    }

    /* Add variation to x-dipole orientations if required. */
    if (element->x_orientation_error_rad > 0.0)
    {
        for (i = 0; i < num_stations; ++i)
            oskar_station_override_element_feed_angle(
                    oskar_telescope_station(t, i),
                    element->seed_x_orientation_error, 1,
                    element->x_orientation_error_rad, 0.0, 0.0, status);
    }

    /* Add variation to y-dipole orientations if required. */
    if (element->y_orientation_error_rad > 0.0)
    {
        for (i = 0; i < num_stations; ++i)
            oskar_station_override_element_feed_angle(
                    oskar_telescope_station(t, i),
                    element->seed_y_orientation_error, 0,
                    element->y_orientation_error_rad, 0.0, 0.0, status);
    }

    /* Apply pointing file override. */
    oskar_telescope_load_pointing_file(t, settings->obs.pointing_file, status);

    return t;
}


static void set_station_data(oskar_Station* station,
        const oskar_Settings_old* s)
{
    int i;
    const oskar_SettingsApertureArray* aa = &s->telescope.aperture_array;
    oskar_station_set_normalise_final_beam(station,
            s->telescope.normalise_beams_at_phase_centre);
    oskar_station_set_enable_array_pattern(station, aa->array_pattern.enable);
    oskar_station_set_normalise_array_pattern(station,
            aa->array_pattern.normalise);
    oskar_station_set_seed_time_variable_errors(station,
            aa->array_pattern.element.seed_time_variable_errors);

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
        oskar_element_set_gaussian_fwhm_rad(element,
                ep->taper.gaussian_fwhm_rad);
    }

    /* Recursively set data for child stations. */
    if (oskar_station_has_child(station))
    {
        int num_elements;
        num_elements = oskar_station_num_elements(station);
        for (i = 0; i < num_elements; ++i)
            set_station_data(oskar_station_child(station, i), s);
    }
}

#ifdef __cplusplus
}
#endif
