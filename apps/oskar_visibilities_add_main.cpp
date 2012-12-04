/*
 * Copyright (c) 2012, The University of Oxford
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

#include <oskar_global.h>
#include <interferometry/oskar_visibilities_read.h>
#include <interferometry/oskar_visibilities_init.h>
#include <interferometry/oskar_visibilities_write.h>
#include <interferometry/oskar_visibilities_copy.h>
#include <utility/oskar_get_error_string.h>
#include <utility/oskar_mem_add.h>

#include "extern/ezOptionParser-0.2.0/ezOptionParser.hpp"

#include <string>
#include <cmath>
#include <iostream>
#include <cfloat>

using namespace ez;
using namespace std;

void set_options(ezOptionParser& opt);
void print_usage(ezOptionParser& opt);
bool check_options(ezOptionParser& opt);
bool isCompatible(const oskar_Visibilities& vis1, const oskar_Visibilities& vis2);
void add_visibilities(oskar_Visibilities& out, const oskar_Visibilities& in1,
        const oskar_Visibilities& in2);
void print_error(int status, const char* message);

int main(int argc, const char** argv)
{
    // Register options =======================================================
    ezOptionParser opt;
    set_options(opt);
    opt.parse(argc, argv);
    if (!check_options(opt))
        return 1;

    // Capture options ========================================================
    string in1_path, in2_path, out_path;
    bool visFirst = ((int)opt.firstArgs.size() == 3) &&
            ((int)opt.lastArgs.size() == 0);
    if (visFirst) {
        in1_path = *opt.firstArgs[1];
        in2_path = *opt.firstArgs[2];
    }
    else {
        in1_path = *opt.lastArgs[0];
        in2_path = *opt.lastArgs[1];
    }
    opt.get("-o")->getString(out_path);

    cout << "in1 = " << in1_path << endl;
    cout << "in2 = " << in2_path << endl;
    cout << "out = " << out_path << endl;

    // Load input data ========================================================
    int status = OSKAR_SUCCESS;
    oskar_Visibilities in1_vis;
    oskar_Visibilities in2_vis;
    oskar_visibilities_read(&in1_vis, in1_path.c_str(), &status);
    oskar_visibilities_read(&in2_vis, in2_path.c_str(), &status);
    if (status != OSKAR_SUCCESS)
        print_error(status, "Failed to read visibility data files.");

    // Sanity checks ==========================================================
    int vis1_n = in1_vis.num_baselines * in1_vis.num_times * in1_vis.num_channels;
    int vis2_n = in2_vis.num_baselines * in2_vis.num_times * in2_vis.num_channels;

    cout << "vis1 num data points = " << vis1_n << endl;
    cout << "vis2 num data points = " << vis2_n << endl;

    // TODO get this to return a more useful message.!
    if (!isCompatible(in1_vis, in2_vis))
    {
        cerr << "ERROR: Input visibility data must match!" << endl;
        return 1;
    }
    // TODO handle cases where they are not compatible but can be concatenated?

    // Create output data =====================================================
    int amp_type = in1_vis.amplitude.type;
    int num_channels = in1_vis.num_channels;
    int num_times = in1_vis.num_times;
    int num_stations = in1_vis.num_stations;
    oskar_Visibilities out_vis;
    oskar_visibilities_init(&out_vis, amp_type, OSKAR_LOCATION_CPU, num_channels,
            num_times, num_stations, &status);
    oskar_visibilities_copy(&out_vis, &in1_vis, &status);
    if (status != OSKAR_SUCCESS)
        print_error(status, "Failed to initialise output visibility structure.");

    // Add ====================================================================
    add_visibilities(out_vis, in1_vis, in2_vis);

    // Write output data ======================================================
    oskar_visibilities_write(&out_vis, 0, out_path.c_str(), &status);
    if (status != OSKAR_SUCCESS)
        print_error(status, "Failed writing output visibility structure to file.");

    return 0;
}

void print_error(int status, const char* message)
{
    cerr << "ERROR[" << status << "] " << message << endl;
    cerr << "REASON = " << oskar_get_error_string(status) << endl;
}

void add_visibilities(oskar_Visibilities& out, const oskar_Visibilities& in1,
        const oskar_Visibilities& in2)
{
    int status = OSKAR_SUCCESS;
    oskar_mem_add(&out.amplitude, &in1.amplitude, &in2.amplitude, &status);
    if (status != OSKAR_SUCCESS)
        print_error(status, "Visibility amplitude addition failed.");
}


bool isCompatible(const oskar_Visibilities& v1, const oskar_Visibilities& v2)
{
    if (v1.num_channels != v2.num_channels)
        return false;
    if (v1.num_times != v2.num_times)
        return false;
    if (v1.num_stations != v2.num_stations)
        return false;
    if (v1.num_baselines != v2.num_baselines)
        return false;
    if (fabs(v1.freq_start_hz - v2.freq_start_hz) > DBL_EPSILON)
        return false;
    if (fabs(v1.freq_inc_hz - v2.freq_inc_hz) > DBL_EPSILON)
        return false;
    if (fabs(v1.channel_bandwidth_hz - v2.channel_bandwidth_hz) > DBL_EPSILON)
        return false;
    if (fabs(v1.time_start_mjd_utc - v2.time_start_mjd_utc) > DBL_EPSILON)
        return false;
    if (fabs(v1.time_inc_seconds - v2.time_inc_seconds) > DBL_EPSILON)
        return false;
    if (fabs(v1.phase_centre_ra_deg - v2.phase_centre_ra_deg) > DBL_EPSILON)
        return false;
    if (fabs(v1.phase_centre_dec_deg - v2.phase_centre_dec_deg) > DBL_EPSILON)
        return false;

    if (v1.amplitude.type != v2.amplitude.type)
        return false;

    return true;
}


void print_usage(ezOptionParser& opt)
{
    string usage;
    opt.getUsage(usage);
    cout << usage;
}


void set_options(ezOptionParser& opt)
{
    opt.overview = string(80, '-') + "\n" +
            "OSKAR application to combine visibility files" +
            "\n" + string(80, '-');
    opt.syntax = "\n  $ oskar_visibilities_add [OPTIONS] input1.vis input2.vis";
    opt.example =
            "  $ oskar_visibilities_add file1.vis file2.vis\n"
            "\n"
            ;
    opt.footer =
            "|" + std::string(80, '-') + "\n"
            "| OSKAR (version " + OSKAR_VERSION_STR + ")\n"
            "| Copyright (C) 2012, The University of Oxford.\n"
            "| This program is free and without warranty.\n"
            "|" + std::string(80, '-') + "\n";
    // add(default, required?, num args, delimiter, msg, flag token(s), ...)
    opt.add("", 0, 0, 0, "Display usage instructions.",
            "-h", "-help", "--help", "--usage");
    opt.add("", 0, 0, 0, "Display the OSKAR version.",
            "-v", "--version");
    opt.add("out.vis", 0, 1, 0, "Output visibility filename (default=out.vis).",
            "-o", "--output");
}

bool check_options(ezOptionParser& opt)
{
    if (opt.isSet("-h")) {
        print_usage(opt);
        return false;
    }

    if (opt.isSet("-v")) {
         cout << "OSKAR version " <<OSKAR_VERSION_STR << endl;
         return false;
    }

    std::vector<std::string> badOptions;
    if(!opt.gotRequired(badOptions)) {
        for(int i=0; i < (int)badOptions.size(); ++i)
            cerr << "ERROR: Missing required option " << badOptions[i] << ".\n\n";
        print_usage(opt);
        return false;
    }

    if(!opt.gotExpected(badOptions)) {
        for(int i=0; i < (int)badOptions.size(); ++i)
        {
            cerr << "ERROR: Got unexpected number of arguments for option ";
            cerr << badOptions[i] << ".\n\n";
        }
        print_usage(opt);
        return false;
    }

    bool visFirst = ((int)opt.firstArgs.size() == 3) &&
            ((int)opt.lastArgs.size() == 0);
    bool visEnd = ((int)opt.firstArgs.size() == 1) &&
            ((int)opt.lastArgs.size() == 2);
    if (!visFirst && !visEnd) {
        print_usage(opt);
        return false;
    }

    return true;
}

