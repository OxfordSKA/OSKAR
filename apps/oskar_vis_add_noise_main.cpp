/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <oskar_settings_load.h>
#include <oskar_settings_log.h>

#include <apps/lib/oskar_set_up_telescope.h>
#include <apps/lib/oskar_OptionParser.h>

#include <oskar_log.h>
#include <oskar_get_error_string.h>
#include <oskar_version_string.h>

#include <oskar_vis.h>
#include <oskar_telescope.h>

#include <oskar_binary.h>

#include <string>
#include <vector>
#include <iostream>

using namespace std;

//-----------------------------------------------------------------------------
void set_options(oskar_OptionParser& opt);
bool check_options(oskar_OptionParser& opt, int argc, char** argv);
void check_error(oskar_Log* log, int status);
//void vis_file_settings(oskar_Log* log, oskar_Vis* vis);
//void log_noise_settings(oskar_Log* log, oskar_Settings* settings);
//-----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    int status = 0;

    // Obtain command line options & arguments.
    oskar_OptionParser opt("oskar_vis_add_noise", oskar_version_string());
    set_options(opt);
    if (!check_options(opt, argc, argv))
        return OSKAR_FAIL;
    string settings_file;
    opt.get("-s")->getString(settings_file);
    vector<string> vis_filename_in = opt.getArgs();
    bool verbose = opt.isSet("-v") ? true : false;
    bool quiet   = opt.isSet("-q") ? true : false;
    bool inplace = opt.isSet("-i") ? true : false;

    // Create the log.
    int file_priority = OSKAR_LOG_MESSAGE;
    int term_priority = OSKAR_LOG_MESSAGE;
    if (quiet) term_priority = OSKAR_LOG_WARNING;
    if (verbose) term_priority = OSKAR_LOG_DEBUG;
    oskar_Log* log = oskar_log_create(file_priority, term_priority);

    oskar_log_set_keep_file(log, false);

    oskar_log_message(log, 'M', 0, "Running binary %s", argv[0]);

    // Load the settings file and telescope model.
    oskar_log_section(log, 'M', "Loading settings file '%s'", settings_file.c_str());
    oskar_Settings settings;
    oskar_settings_load(&settings, 0, settings_file.c_str(), &status);
    if (!settings.interferometer.noise.enable)
    {
        oskar_log_error(log, "Noise addition disabled in the settings.");
        return EXIT_FAILURE;
    }
    // FIXME these are not useful settings to print for this app!
    //oskar_log_settings_interferometer(log, &settings);
    //log_noise_settings(log, &settings);
    // FIXME oskar_set_up_telescope should not be printing log messages!
    oskar_Telescope* tel = oskar_set_up_telescope(&settings, log, &status);
    check_error(log, status);

    // Create list of output vis file names.
    vector<string> vis_filename_out(vis_filename_in.size());
    for (int i = 0; i < (int)vis_filename_out.size(); ++i)
    {
        if (inplace)
            vis_filename_out[i] = vis_filename_in[i];
        else
        {
            string str = vis_filename_in[i];
            str.erase(str.end()-4, str.end());
            vis_filename_out[i] = str + "_noise.vis";
        }
    }

    // Print a summary of what is about to happen.
    oskar_log_line(log, 'D', ' ');
    oskar_log_line(log, 'D', '-');
    oskar_log_value(log, 'D', -1, "Number of visibility files", "%li", vis_filename_in.size());
    for (int i = 0; i < (int)vis_filename_in.size(); ++i)
        oskar_log_message(log, 'D', 1, "%s", vis_filename_in[i].c_str());
    oskar_log_value(log, 'D', -1, "Settings file", "%s", settings_file.c_str());
    oskar_log_value(log, 'D', -1, "Verbose", "%s", (verbose?"true":"false"));
    oskar_log_value(log, 'D', -1, "In place", "%s", (inplace?"true":"false"));
    oskar_log_line(log, 'D', '-');

    // Add uncorrelated noise to each of the visibility files.
    for (int i = 0; i < (int)vis_filename_in.size(); ++i)
    {
        oskar_log_line(log, 'D', ' ');
        oskar_log_value(log, 'D', -1, "Loading visibility file", "%s", vis_filename_in[i].c_str());

        // Load the visibility file
        oskar_Vis* vis = oskar_vis_read(vis_filename_in[i].c_str(), &status);

        // TODO Check that the visibility file was written with normalise
        // beam mode enabled. If not print a warning.
        // TODO Also verify any settings in the vis file against those loaded.
        //vis_file_settings(log, vis);
        oskar_log_value(log, 'D', -1, "No. of baselines", "%i", oskar_vis_num_baselines(vis));
        oskar_log_value(log, 'D', -1, "No. of times", "%i", oskar_vis_num_times(vis));
        oskar_log_value(log, 'D', -1, "No. of channels", "%i", oskar_vis_num_channels(vis));

        // Add noise
        int seed = settings.interferometer.noise.seed;
        oskar_vis_add_system_noise(vis, tel, seed, &status);

        // Write the noisy visibility file.
        oskar_log_value(log, 'D', -1, "Writing visibility file", "%s", vis_filename_out[i].c_str());
        oskar_vis_write(vis, 0, vis_filename_out[i].c_str(), &status);

        oskar_vis_free(vis, &status);

        check_error(log, status);
    }

    oskar_telescope_free(tel, &status);
    oskar_log_free(log);
    return status;
}

void set_options(oskar_OptionParser& opt)
{
    opt.setDescription("Application to add noise to OSKAR binary visibility files.");
    opt.addRequired("OSKAR visibility file(s)...");
    opt.addFlag("-s", "OSKAR settings file (noise settings).", 1, "", true);
    opt.addFlag("-i", "Add noise 'in-place'");
    opt.addFlag("-v", "Verbose logging.");
    opt.addFlag("-q", "Suppress all logging output.");
}

bool check_options(oskar_OptionParser& opt, int argc, char** argv)
{
    if (!opt.check_options(argc, argv))
        return false;
    return true;
}

void check_error(oskar_Log* log, int status)
{
    if (status != OSKAR_SUCCESS) {
        oskar_log_error(log, "code[%i] %s", status, oskar_get_error_string(status));
        exit(status);
    }
}

//void vis_file_settings(oskar_Log* log, oskar_Vis* vis)
//{
//    oskar_Mem* s = oskar_vis_settings(vis);
//    //printf("%s\n", oskar_mem_char(s));
//    //oskar_log_message(log, 0, "Checking settings");
//    // TODO load settings from a string rather than a file! (This isn't
//    // possible with QSettings but should be a feature of the new settings interface)
//}
//
//void log_noise_settings(oskar_Log* log, oskar_Settings* s)
//{
//    const oskar_SettingsSystemNoise* n = &s->interferometer.noise;
//    const oskar_SettingsInterferometer* i = &s->interferometer;
//    int width = 45; // Exactly how this this defined?
//                    // Change to start column?
//
//    // Needs significant logic here to determine what should print
//    // This can be handled automatically by the new settings.
//    oskar_log_line(log, 'o');
//    oskar_log_message(log, 0, "Noise settings");
//    oskar_log_value(log, 1, width, "Enabled", "%s", (n->enable?"true":"false"));
//    oskar_log_value(log, 1, width, "Seed", "%i", n->seed);
//    oskar_log_value(log, 1, width, "Frequency specification", "%i", n->freq.specification);
//    oskar_log_value(log, 1, width, "Value specification", "%i", n->value.specification);
//    oskar_log_line(log, 'o');
//
//}
