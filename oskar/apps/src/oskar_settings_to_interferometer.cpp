/*
 * Copyright (c) 2017-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "apps/oskar_settings_log.h"
#include "apps/oskar_settings_to_interferometer.h"

#include <cstdlib>
#include <cstring>

using namespace std;

oskar_Interferometer* oskar_settings_to_interferometer(oskar::SettingsTree* s,
        oskar_Log* log, int* status)
{
    (void) log;
    if (*status || !s) return 0;
    s->clear_group();

    // Create and set up the interferometer simulator.
    int prec = s->to_int("simulator/double_precision", status) ?
            OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_Interferometer* h = oskar_interferometer_create(prec, status);
    oskar_Log* log_ = oskar_interferometer_log(h);

    // Set simulator settings.
    s->begin_group("simulator");
    oskar_interferometer_set_max_sources_per_chunk(h,
            s->to_int("max_sources_per_chunk", status));
    oskar_interferometer_set_settings_path(h, s->file_name());
    if (!s->to_int("use_gpus", status))
        oskar_interferometer_set_gpus(h, 0, 0, status);
    else
    {
        if (s->starts_with("cuda_device_ids", "all", status))
        {
            oskar_interferometer_set_gpus(h, -1, 0, status);
            if (oskar_interferometer_num_gpus(h) == 0)
                oskar_log_warning(log_, "No GPU capability available.");
        }
        else
        {
            int size = 0;
            const int* ids = s->to_int_list("cuda_device_ids", &size, status);
            if (size > 0)
                oskar_interferometer_set_gpus(h, size, ids, status);
        }
    }
    if (s->starts_with("num_devices", "auto", status))
        oskar_interferometer_set_num_devices(h, -1);
    else
        oskar_interferometer_set_num_devices(h,
                s->to_int("num_devices", status));
    oskar_log_set_keep_file(log_, s->to_int("keep_log_file", status));
    oskar_log_set_file_priority(log_,
            s->to_int("write_status_to_log_file", status) ?
                    OSKAR_LOG_STATUS : OSKAR_LOG_MESSAGE);
    s->end_group();

    // Set sky settings.
    s->begin_group("sky");
    oskar_interferometer_set_horizon_clip(h,
            s->to_int("advanced/apply_horizon_clip", status));
    oskar_interferometer_set_zero_failed_gaussians(h,
            s->to_int("advanced/zero_failed_gaussians", status));
    oskar_interferometer_set_source_flux_range(h,
            s->to_double("common_flux_filter/flux_min", status),
            s->to_double("common_flux_filter/flux_max", status));
    s->end_group();

    // Set interferometer settings.
    // Must be done BEFORE the observation settings!
    s->begin_group("interferometer");
    oskar_interferometer_set_correlation_type(h,
            s->to_string("correlation_type", status), status);
    oskar_interferometer_set_max_times_per_block(h,
            s->to_int("max_time_samples_per_block", status));
    oskar_interferometer_set_max_channels_per_block(h,
            s->to_int("max_channels_per_block", status));
    oskar_interferometer_set_output_vis_file(h,
            s->to_string("oskar_vis_filename", status));
    oskar_interferometer_set_output_measurement_set(h,
            s->to_string("ms_filename", status));
    oskar_interferometer_set_force_polarised_ms(h,
            s->to_int("force_polarised_ms", status));
    oskar_interferometer_set_ignore_w_components(h,
            s->to_int("ignore_w_components", status));
    s->end_group();

    // Set observation settings.
    s->begin_group("observation");
    int num_time_steps = s->to_int("num_time_steps", status);
    double inc_sec = s->to_double("length", status) / num_time_steps;
    oskar_interferometer_set_observation_time(h,
            s->to_double("start_time_utc", status), inc_sec, num_time_steps);
    oskar_interferometer_set_observation_frequency(h,
            s->to_double("start_frequency_hz", status),
            s->to_double("frequency_inc_hz", status),
            s->to_int("num_channels", status));
    s->end_group();

    // Return handle to interferometer simulator.
    s->clear_group();
    return h;
}
