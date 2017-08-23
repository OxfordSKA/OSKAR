/*
 * Copyright (c) 2014-2017, The University of Oxford
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

#include "apps/oskar_option_parser.h"
#include "apps/oskar_app_settings.h"
#include "apps/oskar_settings_log.h"
#include "log/oskar_log.h"
#include "telescope/station/element/oskar_element.h"
#include "utility/oskar_dir.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdio>

using namespace oskar;
using std::string;

static const char app[] = "oskar_fit_element_data";

static string construct_element_pathname(string output_dir,
        int port, int element_type_index, double frequency_hz);

int main(int argc, char** argv)
{
    OptionParser opt(app, oskar_version_string(), oskar_app_settings(app));
    opt.add_settings_options();
    opt.add_flag("-q", "Suppress printing.", false, "--quiet");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;
    const char* settings = opt.get_arg(0);
    int e = 0;

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
        printf("%s\n", s->to_string(opt.get_arg(1), &e));
        SettingsTree::free(s);
        return !e ? 0 : EXIT_FAILURE;
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

    // Write settings to log.
    oskar_log_set_keep_file(log, 0);
    oskar_settings_log(s, log);

    // Get the main settings.
    s->begin_group("element_fit");
    string input_cst_file = s->to_string("input_cst_file", &e);
    string input_scalar_file = s->to_string("input_scalar_file", &e);
    string output_dir = s->to_string("output_directory", &e);
    string pol_type = s->to_string("pol_type", &e);
    // string coordinate_system = s->to_string("coordinate_system", &e);
    int element_type_index = s->to_int("element_type_index", &e);
    double frequency_hz = s->to_double("frequency_hz", &e);
    double average_fractional_error =
            s->to_double("average_fractional_error", &e);
    double average_fractional_error_factor_increase =
            s->to_double("average_fractional_error_factor_increase", &e);
    int ignore_below_horizon = s->to_int("ignore_data_below_horizon", &e);
    int ignore_at_pole = s->to_int("ignore_data_at_pole", &e);
    int port = pol_type == "X" ? 1 : pol_type == "Y" ? 2 : 0;

    // Check that the input and output files have been set.
    if ((input_cst_file.empty() && input_scalar_file.empty()) ||
            output_dir.empty())
    {
        oskar_log_error(log, "Specify input and output file names.");
        SettingsTree::free(s);
        return EXIT_FAILURE;
    }

    // Create an element model.
    oskar_Element* element = oskar_element_create(OSKAR_DOUBLE, OSKAR_CPU, &e);

    // Load the CST text file for the correct port, if specified (X=1, Y=2).
    if (!input_cst_file.empty())
    {
        oskar_log_line(log, 'M', ' ');
        oskar_log_message(log, 'M', 0, "Loading CST element pattern: %s",
                input_cst_file.c_str());
        oskar_element_load_cst(element, log, port, frequency_hz,
                input_cst_file.c_str(), average_fractional_error,
                average_fractional_error_factor_increase,
                ignore_at_pole, ignore_below_horizon, &e);

        // Construct the output file name based on the settings.
        if (port == 0)
        {
            string output = construct_element_pathname(output_dir, 1,
                    element_type_index, frequency_hz);
            oskar_element_write(element, log, output.c_str(), 1,
                    frequency_hz, &e);
            output = construct_element_pathname(output_dir, 2,
                    element_type_index, frequency_hz);
            oskar_element_write(element, log, output.c_str(), 2,
                    frequency_hz, &e);
        }
        else
        {
            string output = construct_element_pathname(output_dir, port,
                    element_type_index, frequency_hz);
            oskar_element_write(element, log, output.c_str(), port,
                    frequency_hz, &e);
        }
    }

    // Load the scalar text file, if specified.
    if (!input_scalar_file.empty())
    {
        oskar_log_message(log, 'M', 0, "Loading scalar element pattern: %s",
                input_scalar_file.c_str());
        oskar_element_load_scalar(element, log, frequency_hz,
                input_scalar_file.c_str(), average_fractional_error,
                average_fractional_error_factor_increase,
                ignore_at_pole, ignore_below_horizon, &e);

        // Construct the output file name based on the settings.
        string output = construct_element_pathname(output_dir, 0,
                element_type_index, frequency_hz);
        oskar_element_write(element, log, output.c_str(), 0,
                frequency_hz, &e);
    }

    // Check for errors.
    if (e) oskar_log_error(log, "Run failed with code %i: %s.", e,
            oskar_get_error_string(e));

    // Free memory.
    oskar_element_free(element, &e);
    oskar_log_free(log);
    SettingsTree::free(s);
    return e;
}


static string construct_element_pathname(string output_dir,
        int port, int element_type_index, double frequency_hz)
{
    std::ostringstream stream;
    stream << "element_pattern_fit_";
    if (port == 0)
        stream << "scalar_";
    else if (port == 1)
        stream << "x_";
    else if (port == 2)
        stream << "y_";

    // Append the element type index.
    stream << element_type_index << "_";

    // Append the frequency in MHz.
    stream << std::fixed << std::setprecision(0)
    << std::setfill('0') << std::setw(3) << frequency_hz / 1.0e6;

    // Append the file extension.
    stream << ".bin";

    // Construct the full path.
    char* path = oskar_dir_get_path(output_dir.c_str(), stream.str().c_str());
    string p = string(path);
    free(path);
    return p;
}

