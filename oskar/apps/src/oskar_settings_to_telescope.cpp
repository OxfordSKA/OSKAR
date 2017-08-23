/*
 * Copyright (c) 2011-2017, The University of Oxford
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

#include "apps/oskar_settings_to_telescope.h"

#include "telescope/oskar_telescope.h"
#include "utility/oskar_get_error_string.h"
#include "log/oskar_log.h"
#include "math/oskar_cmath.h"

#include <limits.h>
#include <cstdio>
#include <cstdlib>

using oskar::SettingsTree;

#define D2R M_PI/180.0

/* Private functions. */
static void set_station_data(oskar_Station* station, SettingsTree* s,
        int* status);

oskar_Telescope* oskar_settings_to_telescope(SettingsTree* s,
        oskar_Log* log, int* status)
{
    if (*status || !s) return 0;
    s->clear_group();

    /* Create an empty telescope model. */
    int type = s->to_int("simulator/double_precision", status) ?
            OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_Telescope* t = oskar_telescope_create(type, OSKAR_CPU, 0, status);

    /************************************************************************/
    /* Options that affect the load. */
    if (s->contains("interferometer"))
        oskar_telescope_set_enable_noise(t,
                s->to_int("interferometer/noise/enable", status),
                s->to_int("interferometer/noise/seed", status));
    oskar_telescope_set_pol_mode(t,
            s->to_string("telescope/pol_mode", status), status);
    oskar_telescope_set_allow_station_beam_duplication(t,
            s->to_int("telescope/allow_station_beam_duplication", status));
    oskar_telescope_set_enable_numerical_patterns(t,
            s->to_int("telescope/aperture_array/element_pattern/"
                    "enable_numerical", status));

    /************************************************************************/
    /* Load telescope model folders to define the stations. */
    oskar_telescope_load(t,
            s->to_string("telescope/input_directory", status), log, status);
    if (*status) return t;

    /* Return if no stations were found. */
    int num_stations = oskar_telescope_num_stations(t);
    if (num_stations < 1)
    {
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE;
        oskar_log_error(log, "Telescope model is empty.");
        return t;
    }

    /************************************************************************/
    /* Set remaining options after the stations have been defined. */
    oskar_telescope_set_station_type(t,
            s->to_string("telescope/station_type", status), status);
    oskar_telescope_set_gaussian_station_beam_width(t,
            s->to_double("telescope/gaussian_beam/fwhm_deg", status),
            s->to_double("telescope/gaussian_beam/ref_freq_hz", status));
    if (s->contains("observation"))
        oskar_telescope_set_phase_centre(t,
                OSKAR_SPHERICAL_TYPE_EQUATORIAL,
                s->to_double("observation/phase_centre_ra_deg", status) * D2R,
                s->to_double("observation/phase_centre_dec_deg", status) * D2R);
    if (s->contains("interferometer"))
    {
        int num_channels = 0;
        double freq_st_hz = 0.0, freq_inc_hz = 0.0;
        if (s->contains("observation"))
        {
            num_channels = s->to_int("observation/num_channels", status);
            freq_st_hz = s->to_double("observation/start_frequency_hz", status);
            freq_inc_hz = s->to_double("observation/frequency_inc_hz", status);
        }
        s->begin_group("interferometer");
        oskar_telescope_set_channel_bandwidth(t,
                s->to_double("channel_bandwidth_hz", status));
        oskar_telescope_set_time_average(t,
                s->to_double("time_average_sec", status));
        oskar_telescope_set_uv_filter(t,
                s->to_double("uv_filter_min", status),
                s->to_double("uv_filter_max", status),
                s->to_string("uv_filter_units", status), status);
        switch (s->first_letter("noise/freq", status))
        {
        case 'R': /* Range. */
            oskar_telescope_set_noise_freq(t,
                    s->to_double("noise/freq/start", status),
                    s->to_double("noise/freq/inc", status),
                    s->to_int("noise/freq/number", status),
                    status);
            break;
        case 'O': /* Observation settings. */
            oskar_telescope_set_noise_freq(t, freq_st_hz, freq_inc_hz,
                    num_channels, status);
            break;
        case 'D': /* Data file. */
            oskar_telescope_set_noise_freq_file(t,
                    s->to_string("noise/freq/file", status), status);
            break;
        }
        switch (s->first_letter("noise/rms", status))
        {
        case 'R': /* Range. */
            oskar_telescope_set_noise_rms(t,
                    s->to_double("noise/rms/start", status),
                    s->to_double("noise/rms/end", status), status);
            break;
        case 'D': /* Data file. */
            oskar_telescope_set_noise_rms_file(t,
                    s->to_string("noise/rms/file", status), status);
            break;
        }
    }
    for (int i = 0; i < num_stations; ++i)
        set_station_data(oskar_telescope_station(t, i), s, status);

    /* Apply element level overrides. */
    s->clear_group();
    s->begin_group("telescope/aperture_array/array_pattern/element");

    /* Override station element systematic/fixed gain errors if required. */
    double gain = s->to_double("gain", status);
    double gain_err_fixed = s->to_double("gain_error_fixed", status);
    if (gain > 0.0 || gain_err_fixed > 0.0)
    {
        int seed = s->to_int("seed_gain_errors", status);
        for (int i = 0; i < num_stations; ++i)
            oskar_station_override_element_gains(
                    oskar_telescope_station(t, i), (unsigned int) seed,
                    gain, gain_err_fixed, status);
    }

    /* Override station element time-variable gain errors if required. */
    double gain_err_time = s->to_double("gain_error_time", status);
    if (gain_err_time > 0.0)
    {
        for (int i = 0; i < num_stations; ++i)
            oskar_station_override_element_time_variable_gains(
                    oskar_telescope_station(t, i), gain_err_time, status);
    }

    /* Override station element systematic/fixed phase errors if required. */
    double phase_err_fixed = s->to_double("phase_error_fixed_deg", status) * D2R;
    if (phase_err_fixed > 0.0)
    {
        int seed = s->to_int("seed_phase_errors", status);
        for (int i = 0; i < num_stations; ++i)
            oskar_station_override_element_phases(
                    oskar_telescope_station(t, i), (unsigned int) seed,
                    phase_err_fixed, status);
    }

    /* Override station element time-variable phase errors if required. */
    double phase_err_time = s->to_double("phase_error_time_deg", status) * D2R;
    if (phase_err_time > 0.0)
    {
        for (int i = 0; i < num_stations; ++i)
            oskar_station_override_element_time_variable_phases(
                    oskar_telescope_station(t, i), phase_err_time, status);
    }

    /* Override station element position errors if required. */
    double position_error_xy_m = s->to_double("position_error_xy_m", status);
    if (position_error_xy_m > 0.0)
    {
        int seed = s->to_int("seed_position_xy_errors", status);
        for (int i = 0; i < num_stations; ++i)
            oskar_station_override_element_xy_position_errors(
                    oskar_telescope_station(t, i), (unsigned int) seed,
                    position_error_xy_m, status);
    }

    /* Add variation to x-dipole orientations if required. */
    double x_rot_err = s->to_double("x_orientation_error_deg", status) * D2R;
    if (x_rot_err > 0.0)
    {
        int seed = s->to_int("seed_x_orientation_error", status);
        for (int i = 0; i < num_stations; ++i)
            oskar_station_override_element_feed_angle(
                    oskar_telescope_station(t, i), (unsigned int) seed, 1,
                    x_rot_err, 0.0, 0.0, status);
    }

    /* Add variation to y-dipole orientations if required. */
    double y_rot_err = s->to_double("y_orientation_error_deg", status) * D2R;
    if (y_rot_err > 0.0)
    {
        int seed = s->to_int("seed_y_orientation_error", status);
        for (int i = 0; i < num_stations; ++i)
            oskar_station_override_element_feed_angle(
                    oskar_telescope_station(t, i), (unsigned int) seed, 0,
                    y_rot_err, 0.0, 0.0, status);
    }

    /* Apply pointing file override. */
    s->clear_group();
    oskar_telescope_load_pointing_file(t,
            s->to_string("observation/pointing_file", status), status);

    return t;
}


void set_station_data(oskar_Station* station, SettingsTree* s, int* status)
{
    if (*status) return;
    s->clear_group();
    oskar_station_set_normalise_final_beam(station,
            s->to_int("telescope/normalise_beams_at_phase_centre", status));
    s->begin_group("telescope/aperture_array/array_pattern");
    oskar_station_set_enable_array_pattern(station,
            s->to_int("enable", status));
    oskar_station_set_normalise_array_pattern(station,
            s->to_int("normalise", status));
    oskar_station_set_seed_time_variable_errors(station,
            (unsigned int) s->to_int(
                    "element/seed_time_variable_errors", status));

    /* Set element pattern data for all element types. */
    s->clear_group();
    s->begin_group("telescope/aperture_array/element_pattern");
    double dipole_length = s->to_double("dipole_length", status);
    char units = s->first_letter("dipole_length_units", status);
    char functional_type = s->first_letter("functional_type", status);
    char taper_type = s->first_letter("taper/type", status);
    double cosine_power = s->to_double("taper/cosine_power", status);
    double fwhm_rad = s->to_double("taper/gaussian_fwhm_deg", status) * D2R;
    for (int i = 0; i < oskar_station_num_element_types(station); ++i)
    {
        oskar_Element* element = oskar_station_element(station, i);
        oskar_element_set_element_type(element, &functional_type, status);
        oskar_element_set_dipole_length(element, dipole_length, &units, status);
        oskar_element_set_taper_type(element, &taper_type, status);
        oskar_element_set_cosine_power(element, cosine_power);
        oskar_element_set_gaussian_fwhm_rad(element, fwhm_rad);
    }

    /* Recursively set data for child stations. */
    if (oskar_station_has_child(station))
    {
        int num_elements = oskar_station_num_elements(station);
        for (int i = 0; i < num_elements; ++i)
            set_station_data(oskar_station_child(station, i), s, status);
    }
}
