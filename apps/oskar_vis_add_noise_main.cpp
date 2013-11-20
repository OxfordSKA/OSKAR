/*
 * Copyright (c) 2013, The University of Oxford
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

#include <apps/lib/oskar_settings_load.h>
#include <apps/lib/oskar_set_up_telescope.h>
#include <apps/lib/oskar_OptionParser.h>

#include <oskar_log.h>
#include <oskar_get_error_string.h>
#include <oskar_version_string.h>

#include <oskar_vis.h>
#include <oskar_telescope.h>

#include <string>
#include <vector>
#include <iostream>

using namespace std;

//------------------------------------------------------------------------------
void set_options(oskar_OptionParser& opt);
bool check_options(oskar_OptionParser& opt, int argc, char** argv);
void check_error(int status);
//------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    int status = OSKAR_SUCCESS;

    // Register options. ======================================================
    oskar_OptionParser opt("oskar_vis_add_noise", oskar_version_string());
    set_options(opt);
    if (!check_options(opt, argc, argv))
        return OSKAR_FAIL;

    // Retrieve options. ======================================================
    string settings_file;
    opt.get("-s")->getString(settings_file);
    vector<string> vis_filename_in = opt.getArgs();
    bool verbose = opt.isSet("-v") ? true : false;
    bool inplace = opt.isSet("-i") ? true : false;

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

    if (verbose)
    {
        cout << endl;
        cout << "---------------------------------------------------------" << endl;
        cout << "Number of visibility files = " << vis_filename_in.size() << endl;
        for (int i = 0; i < (int)vis_filename_in.size(); ++i)
            cout << "  --> " << vis_filename_in[i] << endl;
        cout << "Settings file = " << settings_file << endl;
        cout << "Verbose = " << verbose << endl;
        cout << "In place = " << inplace << endl;
        cout << "---------------------------------------------------------" << endl;
        cout << endl;
    }

    // Add uncorrelated noise. ================================================
    oskar_Settings settings;
    status = oskar_settings_load(&settings, 0, settings_file.c_str());
    if (!settings.interferometer.noise.enable)
    {
        cout << "Warning: Noise addition disabled in the settings." << endl;
        return EXIT_FAILURE;
    }
    check_error(status);
    if (verbose)
    {
        oskar_log_settings_interferometer(0, &settings);
    }

    oskar_Telescope* tel = oskar_set_up_telescope(0, &settings, &status);
    check_error(status);


    for (int i = 0; i < (int)vis_filename_in.size(); ++i)
    {
        if (verbose)
            cout << "Loading visibility file: " << vis_filename_in[i] << endl;
        oskar_Vis* vis = oskar_vis_read(vis_filename_in[i].c_str(), &status);
        check_error(status);

        if (verbose)
        {
            cout << "  No. of baselines: " << oskar_vis_num_baselines(vis) << endl;
            cout << "  No. of times: " << oskar_vis_num_times(vis) << endl;
            cout << "  No. of channels: " << oskar_vis_num_channels(vis) << endl;
        }
        int seed = settings.interferometer.noise.seed;
        oskar_vis_add_system_noise(vis, tel, seed, &status);
        check_error(status);

        if (verbose)
            cout << "Writing visibility file: " << vis_filename_out[i] << endl;
        oskar_vis_write(vis, 0, vis_filename_out[i].c_str(), &status);
        oskar_vis_free(vis, &status);
        check_error(status);
    }

    oskar_telescope_free(tel, &status);
    return status;
}

void set_options(oskar_OptionParser& opt)
{
    opt.setDescription("Application to add noise to OSKAR binary visibility files.");
    opt.addRequired("OSKAR visibility file(s)...");
    opt.addFlag("-s", "OSKAR settings file (noise settings).", 1, "", true);
    opt.addFlag("-i", "Add noise 'in-place'");
    opt.addFlag("-v", "Verbose");
}

bool check_options(oskar_OptionParser& opt, int argc, char** argv)
{
    if (!opt.check_options(argc, argv))
        return false;
    return true;
}

void check_error(int status)
{
    if (status != OSKAR_SUCCESS) {
        cout << "ERROR: code[" << status << "] ";
        cout << oskar_get_error_string(status) << endl;
        exit(status);
    }
}

