/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "apps/oskar_app_settings.h"
#include "apps/oskar_settings_log.h"
#include "apps/oskar_settings_to_beam_pattern.h"
#include "apps/oskar_settings_to_telescope.h"
#include "beam_pattern/oskar_beam_pattern.h"
#include "log/oskar_log.h"
#include "settings/oskar_option_parser.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include <cstdio>
#include <cstdlib>

using namespace oskar;

static const char app[] = "oskar_sim_beam_pattern";

int main(int argc, char** argv)
{
    OptionParser opt(app, oskar_version_string(), oskar_app_settings(app));
    opt.add_settings_options();
    opt.add_flag("-q", "Suppress printing.", false, "--quiet");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;
    const char* settings = opt.get_arg(0);
    int status = 0;

    // Load the settings file.
    SettingsTree* s = oskar_app_settings_tree(app, settings);
    if (!s)
    {
        oskar_log_error(0, "Failed to read settings file '%s'", settings);
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
        if (!ok) oskar_log_error(0, "Failed to set '%s'='%s'", key, val);
        SettingsTree::free(s);
        return ok ? 0 : EXIT_FAILURE;
    }

    // Set up the beam pattern simulator.
    oskar_BeamPattern* sim = oskar_settings_to_beam_pattern(s, NULL, &status);
    oskar_Log* log = oskar_beam_pattern_log(sim);
    int priority = opt.is_set("-q") ? OSKAR_LOG_WARNING : OSKAR_LOG_STATUS;
    oskar_log_set_term_priority(log, priority);

    // Write settings to log.
    oskar_settings_log(s, log);

    // Set up the telescope model.
    oskar_Telescope* tel = oskar_settings_to_telescope(s, log, &status);
    if (!tel || status)
    {
        oskar_log_error(log, "Failed to set up telescope model: %s.",
                oskar_get_error_string(status));
    }
    else
    {
        oskar_beam_pattern_set_telescope_model(sim, tel, &status);
    }
    oskar_telescope_free(tel, &status);

    // Run simulation.
    oskar_beam_pattern_run(sim, &status);

    // Free memory.
    oskar_beam_pattern_free(sim, &status);
    SettingsTree::free(s);
    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}
