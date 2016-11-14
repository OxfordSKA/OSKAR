/*
 * Copyright (c) 2014-2016, The University of Oxford
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
#include <oskar_dir.h>
#include <oskar_get_error_string.h>
#include <oskar_element.h>
#include <oskar_log.h>
#include <oskar_version_string.h>

#include <oskar_settings_old_init.h>
#include <oskar_settings_old_free.h>
#include <oskar_settings_load_element_fit.h>
#include <oskar_settings_load_simulator.h>
#include <oskar_settings_log.h>

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdio>

using std::string;

static void oskar_fit_element_data(const char* settings_file, oskar_Log* log,
        int* status);
static string construct_element_pathname(const char* output_dir,
        int port, int element_type_index, double frequency_hz);

int main(int argc, char** argv)
{
    int error = 0;

    oskar_OptionParser opt("oskar_fit_element_data", oskar_version_string());
    opt.addRequired("settings file");
    opt.addFlag("-q", "Suppress printing.", false, "--quiet");
    if (!opt.check_options(argc, argv))
        return OSKAR_ERR_INVALID_ARGUMENT;

    const char* settings_file = opt.getArg(0);

    // Create the log.
    int file_priority = OSKAR_LOG_MESSAGE;
    int term_priority = opt.isSet("-q") ? OSKAR_LOG_WARNING : OSKAR_LOG_STATUS;
    oskar_Log* log = oskar_log_create(file_priority, term_priority);

    oskar_log_message(log, 'M', 0, "Running binary %s", argv[0]);

    // Do the fitting.
    oskar_fit_element_data(settings_file, log, &error);

    // Check for errors.
    if (error)
    {
        oskar_log_error(log, "Run failed with code %i: %s.", error,
                oskar_get_error_string(error));
    }
    oskar_log_free(log);

    return error;
}


static string construct_element_pathname(const char* output_dir,
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
    char* path = oskar_dir_get_path(output_dir, stream.str().c_str());
    string p = string(path);
    free(path);
    return p;
}


static void oskar_fit_element_data(const char* settings_file, oskar_Log* log,
        int* status)
{
    string output;

    // Load the settings.
    oskar_Settings_old settings;
    oskar_SettingsElementFit *fit = &settings.element_fit;
    oskar_settings_old_init(&settings);
    oskar_settings_load_simulator(&settings.sim, settings_file, status);
    oskar_settings_load_element_fit(fit, settings_file, status);
    if (*status)
    {
        oskar_log_error(log, "Failed to read settings file: %s\n",
                oskar_get_error_string(*status));
        return;
    }

    // Get the main settings.
    oskar_log_set_keep_file(log, settings.sim.keep_log_file);
    oskar_log_settings_element_fit(log, &settings);
    const char* input_cst_file = fit->input_cst_file;
    const char* input_scalar_file = fit->input_scalar_file;
    const char* output_dir = fit->output_directory;
    double frequency_hz = fit->frequency_hz;
    int element_type_index = fit->element_type_index;
    int port = fit->pol_type;

    // Check that the input and output files have been set.
    if ((!input_cst_file && !input_scalar_file) || !output_dir)
    {
        *status = OSKAR_ERR_FILE_IO;
        oskar_settings_old_free(&settings);
        return;
    }

    // Create an element model.
    oskar_Element* element = oskar_element_create(OSKAR_DOUBLE,
            OSKAR_CPU, status);

    // Load the CST text file for the correct port, if specified (X=1, Y=2).
    if (input_cst_file)
    {
        oskar_log_line(log, 'M', ' ');
        oskar_log_message(log, 'M', 0, "Loading CST element pattern: %s",
                input_cst_file);
        oskar_element_load_cst(element, log, port, frequency_hz,
                input_cst_file, fit->average_fractional_error,
                fit->average_fractional_error_factor_increase,
                fit->ignore_data_at_pole, fit->ignore_data_below_horizon,
                status);

        // Construct the output file name based on the settings.
        if (port == 0)
        {
            output = construct_element_pathname(output_dir, 1,
                    element_type_index, frequency_hz);
            oskar_element_write(element, log, output.c_str(), 1,
                    frequency_hz, status);
            output = construct_element_pathname(output_dir, 2,
                    element_type_index, frequency_hz);
            oskar_element_write(element, log, output.c_str(), 2,
                    frequency_hz, status);
        }
        else
        {
            output = construct_element_pathname(output_dir, port,
                    element_type_index, frequency_hz);
            oskar_element_write(element, log, output.c_str(), port,
                    frequency_hz, status);
        }
    }

    // Load the scalar text file, if specified.
    if (input_scalar_file)
    {
        oskar_log_message(log, 'M', 0, "Loading scalar element pattern: %s",
                input_scalar_file);
        oskar_element_load_scalar(element, log, frequency_hz,
                input_scalar_file, fit->average_fractional_error,
                fit->average_fractional_error_factor_increase,
                fit->ignore_data_at_pole, fit->ignore_data_below_horizon,
                status);

        // Construct the output file name based on the settings.
        output = construct_element_pathname(output_dir, 0,
                element_type_index, frequency_hz);
        oskar_element_write(element, log, output.c_str(), 0,
                frequency_hz, status);
    }

    // Free memory.
    oskar_element_free(element, status);
    oskar_settings_old_free(&settings);
}

