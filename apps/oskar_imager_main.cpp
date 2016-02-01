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

#include <apps/lib/oskar_OptionParser.h>

#include <oskar_get_error_string.h>
#include <oskar_imager.h>
#include <oskar_log.h>
#include <oskar_timer.h>
#include <oskar_version_string.h>

#include <oskar_SettingsTree.hpp>
#include <oskar_SettingsDeclareXml.hpp>
#include <oskar_SettingsFileHandlerQSettings.hpp>

#include "settings/xml/oskar_imager_xml_all.h"

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
    int e = 0, end, prec;
    vector<pair<string, string> > failed_keys;

    oskar_OptionParser opt("oskar_imager", oskar_version_string());
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
    settings_declare_xml(&s, oskar_imager_XML_STR);
    SettingsFileHandlerQSettings handler;
    s.set_file_handler(&handler);
    if (!s.load(failed_keys, settings_file)) return OSKAR_ERR_SETTINGS_LOAD;
    for (size_t i = 0; i < failed_keys.size(); ++i)
        oskar_log_warning(log, "Ignoring '%s'='%s'",
                failed_keys[i].first.c_str(), failed_keys[i].second.c_str());

    // TODO Log the relevant settings.
    oskar_log_set_keep_file(log, 0);

    // Create imager and set values from settings.
    s.begin_group("image");
    prec = s.to_int("double_precision", &e) ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_Imager* h = oskar_imager_create(prec, &e);
    oskar_imager_set_ms_column(h, s.to_string("ms_column", &e).c_str(), &e);
    oskar_imager_set_output_root(h, s.to_string("root_path", &e).c_str(), &e);
    oskar_imager_set_image_type(h, s.to_string("image_type", &e).c_str(), &e);
    oskar_imager_set_algorithm(h, s.to_string("algorithm", &e).c_str(), &e);
    oskar_imager_set_fov(h, s.to_double("fov_deg", &e));
    oskar_imager_set_size(h, s.to_int("size", &e));
    end = s.starts_with("channel_end", "max", &e) ? -1 :
            s.to_int("channel_end", &e);
    oskar_imager_set_channel_range(h, s.to_int("channel_start", &e), end,
            s.to_int("channel_snapshots", &e));
    end = s.starts_with("time_end", "max", &e) ? -1 :
            s.to_int("time_end", &e);
    oskar_imager_set_time_range(h, s.to_int("time_start", &e), end,
            s.to_int("time_snapshots", &e));
    oskar_imager_set_grid_kernel(h,
            s.to_string("fft/kernel_type", &e).c_str(),
            s.to_int("fft/support", &e),
            s.to_int("fft/oversample", &e), &e);
    if (s.first_letter("direction", &e) == 'R')
        oskar_imager_set_direction(h,
                s.to_double("direction/ra_deg", &e),
                s.to_double("direction/dec_deg", &e));
    if (!s.starts_with("cuda_device_ids", "all", &e))
    {
        vector<int> ids = s.to_int_list("cuda_device_ids", &e);
        if (ids.size() > 0) oskar_imager_set_gpus(h, ids.size(), &ids[0], &e);
    }

    // Make the images.
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);
    if (!e)
    {
        oskar_log_section(log, 'M', "Starting imager...");
        oskar_timer_resume(tmr);
        oskar_imager_run(h, s.to_string("input_vis_data", &e).c_str(), &e);
    }

    if (!e)
        oskar_log_message(log, 'M', 0, "Imaging completed in %.3f sec.",
                oskar_timer_elapsed(tmr));
    else
        oskar_log_error(log, "Run failed with code %i: %s.", e,
                oskar_get_error_string(e));

    // Free memory.
    oskar_timer_free(tmr);
    oskar_log_free(log);
    oskar_imager_free(h, &e);

    return e;
}
