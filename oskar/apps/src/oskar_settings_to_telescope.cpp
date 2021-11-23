/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "apps/oskar_settings_to_telescope.h"

#include "telescope/oskar_telescope.h"
#include "utility/oskar_get_error_string.h"
#include "math/oskar_cmath.h"

#include <limits.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

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
    {
        oskar_telescope_set_enable_noise(t,
                s->to_int("interferometer/noise/enable", status),
                s->to_int("interferometer/noise/seed", status));
    }
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
    int num_station_models = oskar_telescope_num_station_models(t);
    if (num_station_models < 1)
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
    {
        if (s->starts_with("observation/mode", "Drift", status))
        {
            oskar_telescope_set_phase_centre(t,
                    OSKAR_COORDS_AZEL, 0.0, (M_PI / 2.0));
        }
        else
        {
            oskar_telescope_set_phase_centre(t,
                    OSKAR_COORDS_RADEC,
                    s->to_double("observation/phase_centre_ra_deg", status) * D2R,
                    s->to_double("observation/phase_centre_dec_deg", status) * D2R);
        }
    }
    if (s->contains("interferometer"))
    {
        int num_channels = 0, drift_scan = 0;
        double freq_st_hz = 0.0, freq_inc_hz = 0.0;
        if (s->contains("observation"))
        {
            if (s->starts_with("observation/mode", "Drift", status))
            {
                drift_scan = 1;
            }
            num_channels = s->to_int("observation/num_channels", status);
            freq_st_hz = s->to_double("observation/start_frequency_hz", status);
            freq_inc_hz = s->to_double("observation/frequency_inc_hz", status);
        }
        s->begin_group("interferometer");
        oskar_telescope_set_channel_bandwidth(t,
                s->to_double("channel_bandwidth_hz", status));
        if (!drift_scan)
        {
            oskar_telescope_set_time_average(t,
                    s->to_double("time_average_sec", status));
        }
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
    for (int i = 0; i < num_station_models; ++i)
    {
        set_station_data(oskar_telescope_station(t, i), s, status);
    }

    /* Apply element level overrides. */
    s->clear_group();
    s->begin_group("telescope/aperture_array/array_pattern/element");

    /* Override station element position errors if required. */
    double position_error_xy_m = s->to_double("position_error_xy_m", status);
    if (position_error_xy_m > 0.0)
    {
        int seed = s->to_int("seed_position_xy_errors", status);
        for (int i = 0; i < num_station_models; ++i)
        {
            oskar_station_override_element_xy_position_errors(
                    oskar_telescope_station(t, i),
                    0, (unsigned int) seed, position_error_xy_m, status);
        }
    }

    const char* k_gain[] = {
            "x_gain", "y_gain"};
    const char* k_gain_fixed[] = {
            "x_gain_error_fixed", "y_gain_error_fixed"};
    const char* k_gain_time[] = {
            "x_gain_error_time", "y_gain_error_time"};
    const char* k_seed_gain[] = {
            "seed_x_gain_errors", "seed_y_gain_errors"};
    const char* k_phase_fixed[] = {
            "x_phase_error_fixed_deg", "y_phase_error_fixed_deg"};
    const char* k_phase_time[] = {
            "x_phase_error_time_deg", "y_phase_error_time_deg"};
    const char* k_seed_phase[] = {
            "seed_x_phase_errors", "seed_y_phase_errors"};
    const char* k_cable[] = {
            "x_cable_length_error_m", "y_cable_length_error_m"};
    const char* k_seed_cable[] = {
            "seed_x_cable_length_errors", "seed_y_cable_length_errors"};
    const char* k_rot[] = {
            "x_orientation_error_deg", "y_orientation_error_deg"};
    const char* k_rot_seed[] = {
            "seed_x_orientation_error", "seed_y_orientation_error"};

    for (int feed = 0; feed < 2; feed++)
    {
        /* Override station element systematic/fixed gain errors if required. */
        double gain = s->to_double(k_gain[feed], status);
        double gain_fixed = s->to_double(k_gain_fixed[feed], status);
        if (gain > 0.0 || gain_fixed > 0.0)
        {
            int seed = s->to_int(k_seed_gain[feed], status);
            oskar_telescope_override_element_gains(t,
                    feed, (unsigned int) seed, gain, gain_fixed, status);
        }

        /* Override station element time-variable gain errors if required. */
        double gain_time = s->to_double(k_gain_time[feed], status);
        if (gain_time > 0.0)
        {
            for (int i = 0; i < num_station_models; ++i)
            {
                oskar_station_override_element_time_variable_gains(
                        oskar_telescope_station(t, i),
                        feed, gain_time, status);
            }
        }

        /* Override station element systematic/fixed phase errors if required. */
        double phase_fixed = s->to_double(k_phase_fixed[feed], status) * D2R;
        if (phase_fixed > 0.0)
        {
            int seed = s->to_int(k_seed_phase[feed], status);
            oskar_telescope_override_element_phases(t,
                    feed, (unsigned int) seed, phase_fixed, status);
        }

        /* Override station element time-variable phase errors if required. */
        double phase_time = s->to_double(k_phase_time[feed], status) * D2R;
        if (phase_time > 0.0)
        {
            for (int i = 0; i < num_station_models; ++i)
            {
                oskar_station_override_element_time_variable_phases(
                        oskar_telescope_station(t, i),
                        feed, phase_time, status);
            }
        }

        /* Override station element cable length errors if required. */
        double cable_length_error = s->to_double(k_cable[feed], status);
        if (cable_length_error > 0.0)
        {
            int seed = s->to_int(k_seed_cable[feed], status);
            oskar_telescope_override_element_cable_length_errors(t,
                    feed, (unsigned int) seed, 0.0, cable_length_error, status);
        }

        /* Add variation to element orientations if required. */
        double rot_err = s->to_double(k_rot[feed], status) * D2R;
        if (rot_err > 0.0)
        {
            int seed = s->to_int(k_rot_seed[feed], status);
            for (int i = 0; i < num_station_models; ++i)
            {
                oskar_station_override_element_feed_angle(
                        oskar_telescope_station(t, i),
                        feed, (unsigned int) seed, rot_err, 0.0, 0.0, status);
            }
        }
    }

    /* Set ionosphere parameters if enabled. */
    s->clear_group();
    s->begin_group("telescope");
    const char* ionosphere_screen_type =
            s->to_string("ionosphere_screen_type", status);
    oskar_telescope_set_ionosphere_screen_type(t, ionosphere_screen_type);
    oskar_telescope_set_isoplanatic_screen(t,
            s->to_int("isoplanatic_screen", status));
    if (std::toupper(ionosphere_screen_type[0]) == 'E')
    {
        s->begin_group("external_tec_screen");
        const char* screen_path = s->to_string("input_fits_file", status);
        if (strlen(screen_path) > 0)
        {
            int num_axes = 0;
            int* axis_size = 0;
            double* axis_inc = 0;
            oskar_mem_read_fits(0, 0, 0, screen_path, 0, 0,
                    &num_axes, &axis_size, &axis_inc, status);
            if (!*status)
            {
                oskar_telescope_set_tec_screen_path(t, screen_path);
                oskar_telescope_set_tec_screen_height(t,
                        s->to_double("screen_height_km", status));
                const double pixel_size =
                        s->to_double("screen_pixel_size_m", status);
                /*const double time_interval =
                        s->to_double("screen_time_interval_sec", status);*/
                if (pixel_size > 0.0)
                {
                    oskar_telescope_set_tec_screen_pixel_size(t, pixel_size);
                }
                else
                {
                    oskar_telescope_set_tec_screen_pixel_size(t, axis_inc[0]);
                }
#if 0
                if (time_interval > 0.0)
                {
                    oskar_telescope_set_tec_screen_time_interval(t,
                            time_interval);
                }
                else
#endif
                {
                    oskar_telescope_set_tec_screen_time_interval(t,
                            axis_inc[2]);
                }
            }
            free(axis_size);
            free(axis_inc);
        }
        else
        {
            oskar_log_error(log, "Unable to load ionospheric screen '%s'.",
                    screen_path);
            *status = OSKAR_ERR_FILE_IO;
        }
    }

    /* Apply pointing file override. */
    s->clear_group();
    oskar_telescope_load_pointing_file(t,
            s->to_string("observation/pointing_file", status), status);

    return t;
}


void set_station_data(oskar_Station* station, SettingsTree* s, int* status)
{
    if (*status || !station) return;
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
    oskar_station_set_normalise_element_pattern(station,
            s->to_int("normalise", status));
    oskar_station_set_swap_xy(station, s->to_int("swap_xy", status));
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
        {
            set_station_data(oskar_station_child(station, i), s, status);
        }
    }
}
