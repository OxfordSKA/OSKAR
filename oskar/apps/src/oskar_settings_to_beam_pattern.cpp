/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "apps/oskar_settings_log.h"
#include "apps/oskar_settings_to_beam_pattern.h"

#include <cstdlib>
#include <cstring>

using namespace std;

oskar_BeamPattern* oskar_settings_to_beam_pattern(oskar::SettingsTree* s,
        oskar_Log* log, int* status)
{
    (void) log;
    int size = 0;
    if (*status || !s) return 0;
    s->clear_group();

    // Create simulator and set values from settings.
    int prec = s->to_int("simulator/double_precision", status) ?
            OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_BeamPattern* h = oskar_beam_pattern_create(prec, status);
    oskar_Log* log_ = oskar_beam_pattern_log(h);

    // Set simulator settings.
    s->begin_group("simulator");
    oskar_beam_pattern_set_max_chunk_size(h,
            s->to_int("max_sources_per_chunk", status));
    if (!s->to_int("use_gpus", status))
    {
        oskar_beam_pattern_set_gpus(h, 0, 0, status);
    }
    else
    {
        if (s->starts_with("cuda_device_ids", "all", status))
        {
            oskar_beam_pattern_set_gpus(h, -1, 0, status);
            if (oskar_beam_pattern_num_gpus(h) == 0)
            {
                oskar_log_warning(log_, "No GPU capability available.");
            }
        }
        else
        {
            const int* ids = s->to_int_list("cuda_device_ids", &size, status);
            if (size > 0)
            {
                oskar_beam_pattern_set_gpus(h, size, ids, status);
            }
        }
    }
    if (s->starts_with("num_devices", "auto", status))
    {
        oskar_beam_pattern_set_num_devices(h, -1);
    }
    else
    {
        oskar_beam_pattern_set_num_devices(h, s->to_int("num_devices", status));
    }
    oskar_log_set_keep_file(log_, s->to_int("keep_log_file", status));
    oskar_log_set_file_priority(log_,
            s->to_int("write_status_to_log_file", status) ?
            OSKAR_LOG_STATUS : OSKAR_LOG_MESSAGE);
    s->end_group();

    // Set observation settings.
    s->begin_group("observation");
    int num_time_steps = s->to_int("num_time_steps", status);
    double inc_sec = s->to_double("length", status) / num_time_steps;
    oskar_beam_pattern_set_observation_time(h,
            s->to_double("start_time_utc", status), inc_sec, num_time_steps);
    oskar_beam_pattern_set_observation_frequency(h,
            s->to_double("start_frequency_hz", status),
            s->to_double("frequency_inc_hz", status),
            s->to_int("num_channels", status));
    s->end_group();

    // Set beam pattern options.
    s->begin_group("beam_pattern");
    if (s->to_int("all_stations", status))
    {
        oskar_beam_pattern_set_station_ids(h, -1, 0);
    }
    else
    {
        const int* station_ids = s->to_int_list("station_ids", &size, status);
        oskar_beam_pattern_set_station_ids(h, size, station_ids);
    }
    oskar_beam_pattern_set_coordinate_frame(h,
            s->first_letter("coordinate_frame", status));
    oskar_beam_pattern_set_coordinate_type(h,
            s->first_letter("coordinate_type", status));
    const int* image_size = s->to_int_list("beam_image/size", &size, status);
    if (size == 1)
    {
        oskar_beam_pattern_set_image_size(h, image_size[0], image_size[0]);
    }
    else if (size > 1)
    {
        oskar_beam_pattern_set_image_size(h, image_size[0], image_size[1]);
    }
    const int specify_cellsize =
            s->to_int("beam_image/specify_cellsize", status);
    if (specify_cellsize)
    {
        oskar_beam_pattern_set_image_cellsize(h,
                s->to_double("beam_image/cellsize_arcsec", status));
    }
    else
    {
        const double* image_fov =
                s->to_double_list("beam_image/fov_deg", &size, status);
        if (size == 1)
        {
            oskar_beam_pattern_set_image_fov(h, image_fov[0], image_fov[0]);
        }
        else if (size > 1)
        {
            oskar_beam_pattern_set_image_fov(h, image_fov[0], image_fov[1]);
        }
    }
    oskar_beam_pattern_set_root_path(h,
            s->to_string("root_path", status));
    oskar_beam_pattern_set_sky_model_file(h,
            s->to_string("sky_model/file", status));
    s->end_group();

    // Set output options.
    s->begin_group("beam_pattern/output");
    oskar_beam_pattern_set_average_single_axis(h,
            s->first_letter("average_single_axis", status));
    oskar_beam_pattern_set_average_time_and_channel(h,
            s->to_int("average_time_and_channel", status));
    oskar_beam_pattern_set_separate_time_and_channel(h,
            s->to_int("separate_time_and_channel", status));
    s->end_group();

    // Set output files.
    s->begin_group("beam_pattern");
    oskar_beam_pattern_set_auto_power_fits(h,
            s->to_int("station_outputs/fits_image/auto_power", status));
    oskar_beam_pattern_set_auto_power_phase_fits(h,
            s->to_int("station_outputs/fits_image/auto_power_phase", status));
    oskar_beam_pattern_set_auto_power_real_fits(h,
            s->to_int("station_outputs/fits_image/auto_power_real", status));
    oskar_beam_pattern_set_auto_power_imag_fits(h,
            s->to_int("station_outputs/fits_image/auto_power_imag", status));
    oskar_beam_pattern_set_auto_power_text(h,
            s->to_int("station_outputs/text_file/auto_power", status));
    oskar_beam_pattern_set_voltage_amp_fits(h,
            s->to_int("station_outputs/fits_image/amp", status));
    oskar_beam_pattern_set_voltage_amp_text(h,
            s->to_int("station_outputs/text_file/amp", status));
    oskar_beam_pattern_set_voltage_phase_fits(h,
            s->to_int("station_outputs/fits_image/phase", status));
    oskar_beam_pattern_set_voltage_phase_text(h,
            s->to_int("station_outputs/text_file/phase", status));
    oskar_beam_pattern_set_voltage_raw_text(h,
            s->to_int("station_outputs/text_file/raw_complex", status));
    oskar_beam_pattern_set_cross_power_amp_fits(h,
            s->to_int("telescope_outputs/fits_image/cross_power_amp", status));
    oskar_beam_pattern_set_cross_power_amp_text(h,
            s->to_int("telescope_outputs/text_file/cross_power_amp", status));
    oskar_beam_pattern_set_cross_power_phase_fits(h,
            s->to_int("telescope_outputs/fits_image/cross_power_phase",
                    status));
    oskar_beam_pattern_set_cross_power_phase_text(h,
            s->to_int("telescope_outputs/text_file/cross_power_phase", status));
    oskar_beam_pattern_set_cross_power_real_fits(h,
            s->to_int("telescope_outputs/fits_image/cross_power_real", status));
    oskar_beam_pattern_set_cross_power_imag_fits(h,
            s->to_int("telescope_outputs/fits_image/cross_power_imag", status));
    oskar_beam_pattern_set_cross_power_raw_text(h,
            s->to_int("telescope_outputs/text_file/cross_power_raw_complex",
                    status));
    s->end_group();

    // Set test source configuration.
    s->begin_group("beam_pattern/test_source");
    oskar_beam_pattern_set_test_source_stokes_i(h,
            s->to_int("stokes_i", status));
    oskar_beam_pattern_set_test_source_stokes_custom(h,
            s->to_int("custom", status),
            s->to_double("custom_stokes_i", status),
            s->to_double("custom_stokes_q", status),
            s->to_double("custom_stokes_u", status),
            s->to_double("custom_stokes_v", status), status);
    s->end_group();

    // Return handle to beam pattern simulator.
    s->clear_group();
    return h;
}
