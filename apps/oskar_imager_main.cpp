/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "apps/oskar_app_settings.h"
#include "apps/oskar_settings_log.h"
#include "apps/oskar_settings_to_imager.h"
#include "imager/oskar_imager.h"
#include "log/oskar_log.h"
#include "settings/oskar_option_parser.h"
#include "utility/oskar_version_string.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

    // Ensure the images are going somewhere.
    if (!strlen(s->to_string("image/root_path", &status)))
    {
        int n = 0;
        const char* const* files =
                s->to_string_list("image/input_vis_data", &n, &status);
        if (n == 0 || !files || !files[0] || !strlen(files[0]))
        {
            oskar_log_error(0, "No input file or output file has been set.");
            SettingsTree::free(s);
            return EXIT_FAILURE;
        }
        const char* ptr = strrchr(files[0], '.');
        std::string fname(files[0], ptr ? (ptr - files[0]) : strlen(files[0]));
        s->set_value("image/root_path", fname.c_str(), false);
    }

    // Set up the imager.
    oskar_Imager* imager = oskar_settings_to_imager(s, NULL, &status);
    int priority = opt.is_set("-q") ? OSKAR_LOG_WARNING : OSKAR_LOG_STATUS;
    oskar_log_set_term_priority(oskar_imager_log(imager), priority);

    // Log settings.
    oskar_settings_log(s, oskar_imager_log(imager));

    // Make the images.
    oskar_imager_run(imager, 0, 0, 0, 0, &status);

    // Free memory.
    oskar_imager_free(imager, &status);
    SettingsTree::free(s);
    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}
