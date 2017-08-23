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

#include "apps/oskar_app_settings.h"
#include "apps/oskar_option_parser.h"
#include "apps/oskar_settings_log.h"
#include "apps/oskar_settings_to_imager.h"
#include "imager/oskar_imager.h"
#include "log/oskar_log.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace oskar;

static const char app[] = "oskar_imager";

int main(int argc, char** argv)
{
    OptionParser opt(app, oskar_version_string(), oskar_app_settings(app));
    opt.add_settings_options();
    opt.add_flag("-q", "Suppress printing.", false, "--quiet");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;
    const char* settings = opt.get_arg(0);
    int status = 0;

    // Create the log if necessary.
    oskar_Log* log = 0;
    if (!opt.is_set("--get") && !opt.is_set("--set"))
    {
        int priority = opt.is_set("-q") ? OSKAR_LOG_WARNING : OSKAR_LOG_STATUS;
        log = oskar_log_create(OSKAR_LOG_MESSAGE, priority);
        oskar_log_message(log, 'M', 0, "Running binary %s", argv[0]);
        oskar_log_section(log, 'M', "Loading settings file '%s'", settings);
    }

    // Load the settings file.
    SettingsTree* s = oskar_app_settings_tree(app, settings);
    if (!s)
    {
        oskar_log_error(log, "Failed to read settings file.");
        if (log) oskar_log_free(log);
        return EXIT_FAILURE;
    }

    // Get/set setting if necessary.
    if (opt.is_set("--get"))
    {
        printf("%s\n", s->to_string(opt.get_arg(1), &status));
        SettingsTree::free(s);
        return !status ? 0 : EXIT_FAILURE;
    }
    else if (opt.is_set("--set"))
    {
        const char* key = opt.get_arg(1);
        const char* val = opt.get_arg(2);
        bool ok = val ? s->set_value(key, val) : s->set_default(key);
        if (!ok) oskar_log_error(log, "Failed to set '%s'='%s'", key, val);
        SettingsTree::free(s);
        return ok ? 0 : EXIT_FAILURE;
    }

    // Ensure the images are going somewhere.
    if (!strlen(s->to_string("image/root_path", &status)))
    {
        int n = 0;
        const char* const* files =
                s->to_string_list("image/input_vis_data", &n, &status);
        if (n == 0 || !files || !files[0] || !strlen(files[0]))
        {
            oskar_log_error(log, "No input file or output file has been set.");
            if (log) oskar_log_free(log);
            SettingsTree::free(s);
            return EXIT_FAILURE;
        }
        const char* ptr = strrchr(files[0], '.');
        std::string fname(files[0], ptr ? (ptr - files[0]) : strlen(files[0]));
        s->set_value("image/root_path", fname.c_str(), false);
    }

    // Write settings to log.
    oskar_log_set_keep_file(log, 0);
    oskar_settings_log(s, log);

    // Set up the imager.
    oskar_Imager* imager = oskar_settings_to_imager(s, log, &status);

    // Make the images.
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);
    if (imager && !status)
    {
        oskar_log_section(log, 'M', "Starting imager...");
        oskar_timer_resume(tmr);
        oskar_imager_run(imager, 0, 0, 0, 0, &status);
    }

    // Check for errors.
    if (!status)
        oskar_log_message(log, 'M', 0, "Run completed in %.3f sec.",
                oskar_timer_elapsed(tmr));
    else
        oskar_log_error(log, "Run failed with code %i: %s.", status,
                oskar_get_error_string(status));

    // Free memory.
    oskar_timer_free(tmr);
    oskar_imager_free(imager, &status);
    oskar_log_free(log);
    SettingsTree::free(s);
    return status;
}
