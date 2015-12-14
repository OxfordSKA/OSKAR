/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_vis.h>
#include <oskar_get_error_string.h>
#include <oskar_version_string.h>
#include <apps/lib/oskar_OptionParser.h>

#include <string>
#include <cmath>
#include <iostream>
#include <cfloat>
#include <vector>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
static void set_options(oskar_OptionParser& opt);
static bool check_options(oskar_OptionParser& opt, int argc, char** argv);
static bool isCompatible(const oskar_Vis* vis1, const oskar_Vis* vis2);
static void print_error(int status, const char* message);
// -----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    // Register options =======================================================
    oskar_OptionParser opt("oskar_vis_add", oskar_version_string());
    set_options(opt);
    if (!check_options(opt, argc, argv))
        return OSKAR_FAIL;

    // Retrieve options ========================================================
    string out_path;
    opt.get("-o")->getString(out_path);
    vector<string> in_files = opt.getInputFiles(2);
    bool verbose = opt.isSet("-q") ? false : true;
    int num_in_files = in_files.size();

    // Print if verbose.
    if (verbose)
    {
        cout << "Output visibility file: " << out_path << endl;
        cout << "Combining the " << num_in_files << " input files:" << endl;
        for (int i = 0; i < num_in_files; ++i)
        {
            cout << "  [" << setw(2) << i << "] " << in_files[i] << endl;
        }
    }

    // Add the data. ==========================================================
    int status = 0;

    // Load the first visibility structure.
    oskar_Binary* h;
    h = oskar_binary_create(in_files[0].c_str(), 'r', &status);
    oskar_Vis* out = oskar_vis_read(h, &status);
    oskar_binary_free(h);
    if (status)
    {
        string msg = "Failed to read visibility data file " + in_files[0];
        print_error(status, msg.c_str());
    }
    oskar_mem_clear_contents(oskar_vis_settings_path(out), &status);
    // TODO write some sort of tag into here to indicate this is an
    // accumulated visibility data set...

    // Loop over other visibility files and combine.
    for (int i = 1; i < num_in_files; ++i)
    {
        if (status) break;

        h = oskar_binary_create(in_files[i].c_str(), 'r', &status);
        oskar_Vis* in = oskar_vis_read(h, &status);
        oskar_binary_free(h);
        if (status)
        {
            string msg = "Failed to read visibility data file " + in_files[i];
            print_error(status, msg.c_str());
            break;
        }
        if (!isCompatible(out, in))
        {
            cerr << "ERROR: Input visibility data must match!" << endl;
            status = OSKAR_ERR_TYPE_MISMATCH;
        }
        oskar_mem_add(oskar_vis_amplitude(out), oskar_vis_amplitude_const(out),
                oskar_vis_amplitude_const(in),
                oskar_mem_length(oskar_vis_amplitude(out)), &status);
        if (status)
            print_error(status, "Visibility amplitude addition failed.");
        oskar_vis_free(in, &status);
    }

    // Write output data ======================================================
    if (verbose)
        cout << "Writing OSKAR visibility file: " << out_path << endl;
    oskar_vis_write(out, 0, out_path.c_str(), &status);
    oskar_vis_free(out, &status);
    if (status)
        print_error(status, "Failed writing output visibility structure to file.");

    return status;
}

static void print_error(int status, const char* message)
{
    cerr << "ERROR[" << status << "] " << message << endl;
    cerr << "REASON: " << oskar_get_error_string(status) << endl;
}


static bool isCompatible(const oskar_Vis* v1, const oskar_Vis* v2)
{
    if (oskar_vis_num_channels(v1) != oskar_vis_num_channels(v2))
        return false;
    if (oskar_vis_num_times(v1) != oskar_vis_num_times(v2))
        return false;
    if (oskar_vis_num_stations(v1) != oskar_vis_num_stations(v2))
        return false;
    if (oskar_vis_num_baselines(v1) != oskar_vis_num_baselines(v2))
        return false;
    if (fabs(oskar_vis_freq_start_hz(v1) -
            oskar_vis_freq_start_hz(v2)) > DBL_EPSILON)
        return false;
    if (fabs(oskar_vis_freq_inc_hz(v1) -
            oskar_vis_freq_inc_hz(v2)) > DBL_EPSILON)
        return false;
    if (fabs(oskar_vis_channel_bandwidth_hz(v1) -
            oskar_vis_channel_bandwidth_hz(v2)) > DBL_EPSILON)
        return false;
    if (fabs(oskar_vis_time_start_mjd_utc(v1) -
            oskar_vis_time_start_mjd_utc(v2)) > DBL_EPSILON)
        return false;
    if (fabs(oskar_vis_time_inc_sec(v1) -
            oskar_vis_time_inc_sec(v2)) > DBL_EPSILON)
        return false;
    if (fabs(oskar_vis_phase_centre_ra_deg(v1) -
            oskar_vis_phase_centre_ra_deg(v2)) > DBL_EPSILON)
        return false;
    if (fabs(oskar_vis_phase_centre_dec_deg(v1) -
            oskar_vis_phase_centre_dec_deg(v2)) > DBL_EPSILON)
        return false;

    if (oskar_mem_type(oskar_vis_amplitude_const(v1)) !=
            oskar_mem_type(oskar_vis_amplitude_const(v2)))
        return false;

    return true;
}

static void set_options(oskar_OptionParser& opt)
{
    opt.setDescription("Application to combine OSKAR binary visibility files.");
    opt.addRequired("OSKAR visibility files...");
    opt.addFlag("-o", "Output visibility file name", 1, "out.vis", false, "--output");
    opt.addFlag("-q", "Disable log messages", false, "--quiet");
    opt.addExample("oskar_vis_add file1.vis file2.vis");
    opt.addExample("oskar_vis_add file1.vis file2.vis -o combined.vis");
    opt.addExample("oskar_vis_add -q file1.vis file2.vis file3.vis");
    opt.addExample("oskar_vis_add *.vis");
}

static bool check_options(oskar_OptionParser& opt, int argc, char** argv)
{
    if (!opt.check_options(argc, argv))
        return false;
    bool visFirst = ((int)opt.firstArgs.size() >= 3) &&
            ((int)opt.lastArgs.size() == 0);
    bool visEnd = ((int)opt.firstArgs.size() == 1) &&
            ((int)opt.lastArgs.size() >= 2);
    if (!visFirst && !visEnd)
    {
        opt.error("Please provide 2 or more visibility files to combine.");
        return false;
    }
    return true;
}

