/*
 * Copyright (c) 2012-2017, The University of Oxford
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

#include "apps/oskar_OptionParser.h"
#include "apps/oskar_settings_log.h"
#include "apps/oskar_settings_to_sky.h"
#include "apps/oskar_settings_to_telescope.h"
#include "log/oskar_log.h"
#include "settings/oskar_SettingsTree.h"
#include "settings/oskar_SettingsDeclareXml.h"
#include "settings/oskar_SettingsFileHandlerIni.h"
#include "simulator/oskar_simulator.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include "apps/xml/oskar_sim_interferometer_xml_all.h"

#include <cstdio>
#include <cstdlib>

using namespace oskar;
using std::vector;
using std::string;
using std::pair;

static const char settings_def[] = oskar_sim_interferometer_XML_STR;

int main(int argc, char** argv)
{
    OptionParser opt("oskar_sim_interferometer", oskar_version_string(),
            settings_def);
    opt.add_settings_options();
    opt.add_flag("-q", "Suppress printing.", false, "--quiet");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;
    const char* settings_file = opt.get_arg(0);
    int e = 0;

    // Declare settings.
    SettingsTree s;
    SettingsFileHandlerIni handler("oskar_sim_interferometer",
            oskar_version_string());
    settings_declare_xml(&s, settings_def);
    s.set_file_handler(&handler);

    // Create the log if necessary.
    oskar_Log* log = 0;
    if (!opt.is_set("--get") && !opt.is_set("--set"))
    {
        int priority = opt.is_set("-q") ? OSKAR_LOG_WARNING : OSKAR_LOG_STATUS;
        log = oskar_log_create(OSKAR_LOG_MESSAGE, priority);
        oskar_log_message(log, 'M', 0, "Running binary %s", argv[0]);
        oskar_log_section(log, 'M', "Loading settings file '%s'", settings_file);
    }

    // Load the settings file.
    vector<pair<string, string> > failed_keys;
    if (!s.load(settings_file, failed_keys))
    {
        oskar_log_error(log, "Failed to read settings file.");
        if (log) oskar_log_free(log);
        return EXIT_FAILURE;
    }

    // Get/set setting if necessary.
    if (opt.is_set("--get"))
    {
        printf("%s\n", s.to_string(opt.get_arg(1), &e).c_str());
        return !e ? 0 : EXIT_FAILURE;
    }
    else if (opt.is_set("--set"))
    {
        const char* key = opt.get_arg(1);
        const char* val = opt.get_arg(2);
        bool ok = val ? s.set_value(key, val) : s.set_default(key);
        if (!ok) oskar_log_error(log, "Failed to set '%s'='%s'", key, val);
        return ok ? 0 : EXIT_FAILURE;
    }

    // Write settings to log.
    oskar_settings_log(&s, log, failed_keys);

    // Set up the sky model.
    oskar_Sky* sky = oskar_settings_to_sky(&s, log, &e);
    if (!sky || e)
    {
        oskar_log_error(log, "Failed to set up sky model: %s.",
                oskar_get_error_string(e));
        oskar_sky_free(sky, &e);
        oskar_log_free(log);
        return e;
    }

    // Set up the telescope model.
    oskar_Telescope* tel = oskar_settings_to_telescope(&s, log, &e);
    if (!tel || e)
    {
        oskar_log_error(log, "Failed to set up telescope model: %s.",
                oskar_get_error_string(e));
        oskar_telescope_free(tel, &e);
        oskar_sky_free(sky, &e);
        oskar_log_free(log);
        return e;
    }

    // Create simulator and set values from settings.
    s.begin_group("simulator");
    int prec = s.to_int("double_precision", &e) ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int max_sources_per_chunk = s.to_int("max_sources_per_chunk", &e);
    oskar_Simulator* h = oskar_simulator_create(prec, &e);
    oskar_simulator_set_log(h, log);
    oskar_simulator_set_settings_path(h, settings_file);
    if (!s.to_int("use_gpus", &e))
        oskar_simulator_set_gpus(h, 0, 0, &e);
    else
    {
        if (s.starts_with("cuda_device_ids", "all", &e))
            oskar_simulator_set_gpus(h, -1, 0, &e);
        else
        {
            vector<int> ids = s.to_int_list("cuda_device_ids", &e);
            if (ids.size() > 0)
                oskar_simulator_set_gpus(h, ids.size(), &ids[0], &e);
        }
    }
    if (s.starts_with("num_devices", "auto", &e))
        oskar_simulator_set_num_devices(h, -1);
    else
        oskar_simulator_set_num_devices(h, s.to_int("num_devices", &e));
    oskar_log_set_keep_file(log, s.to_int("keep_log_file", &e));
    oskar_log_set_file_priority(log, s.to_int("write_status_to_log_file", &e) ?
            OSKAR_LOG_STATUS : OSKAR_LOG_MESSAGE);
    s.end_group();

    // Set sky settings.
    s.begin_group("sky");
    oskar_simulator_set_horizon_clip(h,
            s.to_int("advanced/apply_horizon_clip", &e));
    oskar_simulator_set_zero_failed_gaussians(h,
            s.to_int("advanced/zero_failed_gaussians", &e));
    oskar_simulator_set_source_flux_range(h,
            s.to_double("common_flux_filter/flux_min", &e),
            s.to_double("common_flux_filter/flux_max", &e));
    s.end_group();

    // Set observation settings.
    s.begin_group("observation");
    int num_time_steps = s.to_int("num_time_steps", &e);
    double inc_sec = s.to_double("length", &e) / num_time_steps;
    oskar_simulator_set_observation_time(h, s.to_double("start_time_utc", &e),
            inc_sec, num_time_steps);
    oskar_simulator_set_observation_frequency(h,
            s.to_double("start_frequency_hz", &e),
            s.to_double("frequency_inc_hz", &e),
            s.to_int("num_channels", &e));
    s.end_group();

    // Set interferometer settings.
    s.begin_group("interferometer");
    oskar_simulator_set_correlation_type(h,
            s.to_string("correlation_type", &e).c_str(), &e);
    oskar_simulator_set_max_times_per_block(h,
            s.to_int("max_time_samples_per_block", &e));
    oskar_simulator_set_output_vis_file(h,
            s.to_string("oskar_vis_filename", &e).c_str());
#ifndef OSKAR_NO_MS
    oskar_simulator_set_output_measurement_set(h,
            s.to_string("ms_filename", &e).c_str());
    oskar_simulator_set_force_polarised_ms(h,
            s.to_int("force_polarised_ms", &e));
#endif
    s.end_group();

    // Set the sky model and telescope model.
    oskar_simulator_set_sky_model(h, sky, max_sources_per_chunk, &e);
    oskar_simulator_set_telescope_model(h, tel, &e);

    // Run simulation.
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);
    oskar_timer_resume(tmr);
    oskar_simulator_run(h, &e);

    // Check for errors.
    if (!e)
        oskar_log_message(log, 'M', 0, "Simulation completed in %.3f sec.",
                oskar_timer_elapsed(tmr));
    else
        oskar_log_error(log, "Run failed with code %i: %s.", e,
                oskar_get_error_string(e));

    // Free memory.
    oskar_sky_free(sky, &e);
    oskar_telescope_free(tel, &e);
    oskar_timer_free(tmr);
    oskar_simulator_free(h, &e);
    oskar_log_free(log);

    return e;
}
