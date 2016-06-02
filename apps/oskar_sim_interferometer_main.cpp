/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#include <apps/lib/oskar_sim_interferometer.h>
#include <apps/lib/oskar_OptionParser.h>

#include <oskar_get_error_string.h>
#include <oskar_log.h>
#include <oskar_set_up_sky.h>
#include <oskar_set_up_telescope.h>
#include <oskar_timer.h>
#include <oskar_version_string.h>

#include <oskar_settings_load.h>
#include <oskar_settings_log.h>
#include <oskar_SettingsTree.hpp>
#include <oskar_SettingsDeclareXml.hpp>
#include <oskar_SettingsFileHandlerQSettings.hpp>

#include "settings/xml/oskar_sim_interferometer_xml_all.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace oskar;
using std::vector;
using std::string;
using std::pair;

int main(int argc, char** argv)
{
    int e = 0;
    vector<pair<string, string> > failed_keys;

    oskar_OptionParser opt("oskar_sim_interferometer", oskar_version_string());
    opt.addRequired("settings file");
    opt.addFlag("-q", "Suppress printing.", false, "--quiet");
    if (!opt.check_options(argc, argv)) return OSKAR_ERR_INVALID_ARGUMENT;
    const char* settings_file = opt.getArg(0);

    // Create the log.
    int file_priority = OSKAR_LOG_MESSAGE;
    int term_priority = opt.isSet("-q") ? OSKAR_LOG_WARNING : OSKAR_LOG_STATUS;
    oskar_Log* log = oskar_log_create(file_priority, term_priority);
    oskar_log_message(log, 'M', 0, "Running binary %s", argv[0]);

    // Load the settings file.
    oskar_log_section(log, 'M', "Loading settings file '%s'", settings_file);
    SettingsTree s;
    settings_declare_xml(&s, oskar_sim_interferometer_XML_STR);
    SettingsFileHandlerQSettings handler;
    s.set_file_handler(&handler);
    if (!s.load(failed_keys, settings_file)) return OSKAR_ERR_SETTINGS_LOAD;
    for (size_t i = 0; i < failed_keys.size(); ++i)
        oskar_log_warning(log, "Ignoring '%s'='%s'",
                failed_keys[i].first.c_str(), failed_keys[i].second.c_str());

    // Log the relevant settings. (TODO fix/automate these functions)
    oskar_Settings_old s_old;
    oskar_settings_old_load(&s_old, log, settings_file, &e);
    oskar_log_settings_simulator(log, &s_old);
    oskar_log_settings_sky(log, &s_old);
    oskar_log_settings_observation(log, &s_old);
    oskar_log_settings_telescope(log, &s_old);
    oskar_log_settings_interferometer(log, &s_old);

    // Set up sky model and telescope model.
    int num_chunks = 0;
    oskar_Sky** sky_chunks = oskar_set_up_sky(&s_old, log, &num_chunks, &e);
    oskar_Telescope *tel = oskar_set_up_telescope(&s_old, log, &e);

    // Create simulator and set values from settings.
    s.begin_group("simulator");
    int prec = s.to_int("double_precision", &e) ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_Simulator* h = oskar_simulator_create(prec, &e);
    oskar_simulator_set_log(h, log);
    oskar_simulator_set_settings_path(h, settings_file);
    oskar_simulator_set_telescope_model(h, tel);
    oskar_simulator_set_sky_chunks(h, num_chunks, sky_chunks);
    oskar_simulator_set_max_sources_per_chunk(h,
            s.to_int("max_sources_per_chunk", &e));
    if (!s.starts_with("cuda_device_ids", "all", &e))
    {
        vector<int> ids = s.to_int_list("cuda_device_ids", &e);
        if (ids.size() > 0)
            oskar_simulator_set_gpus(h, ids.size(), &ids[0], &e);
    }
    oskar_log_set_keep_file(log, s.to_int("keep_log_file", &e));
    oskar_log_set_file_priority(log, s.to_int("write_status_to_log_file", &e) ?
            OSKAR_LOG_STATUS : OSKAR_LOG_MESSAGE);
    s.end_group();

    // Set sky settings.
    s.begin_group("sky");
    oskar_simulator_set_horizon_clip(h,
            s.to_int("advanced/apply_horizon_clip", &e));
    /* FIXME oskar_simulator_set_source_flux_range(h,
            s.to_double("common_flux_filter_min_jy", &e),
            s.to_double("common_flux_filter_max_jy", &e));*/
    s.end_group();

    // Set observation settings.
    s.begin_group("observation");
    int num_time_steps = s.to_int("num_time_steps", &e);
    double inc_sec = s.to_double("length", &e) / num_time_steps;
    oskar_simulator_set_time(h, s.to_double("start_time_utc", &e),
            inc_sec, num_time_steps);
    oskar_simulator_set_frequency(h, s.to_double("start_frequency_hz", &e),
            s.to_double("frequency_inc_hz", &e), s.to_int("num_channels", &e));
    s.end_group();

    // Set interferometer settings.
    s.begin_group("interferometer");
    oskar_simulator_set_correlation_type(h,
            s.to_string("correlation_type", &e).c_str(), &e);
    oskar_simulator_set_max_times_per_block(h,
            s.to_int("max_time_samples_per_block", &e));
    oskar_simulator_set_thermal_noise(h, s.to_int("noise/enable", &e),
            s.to_int("noise/seed", &e));
    oskar_simulator_set_output_vis_file(h,
            s.to_string("oskar_vis_filename", &e).c_str());
#ifndef OSKAR_NO_MS
    oskar_simulator_set_output_measurement_set(h,
            s.to_string("ms_filename", &e).c_str());
    oskar_simulator_set_force_polarised_ms(h,
            s.to_int("force_polarised_ms", &e));
#endif
    s.end_group();

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
    oskar_telescope_free(tel, &e);
    oskar_timer_free(tmr);
    oskar_simulator_free(h, &e);
    oskar_log_free(log);
    for (int i = 0; i < num_chunks; ++i)
        oskar_sky_free(sky_chunks[i], &e);
    free(sky_chunks);

    return e;
}
