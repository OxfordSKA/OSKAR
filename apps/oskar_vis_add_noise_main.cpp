/*
 * Copyright (c) 2013-2017, The University of Oxford
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
#include "apps/oskar_settings_to_telescope.h"
#include "binary/oskar_binary.h"
#include "log/oskar_log.h"
#include "telescope/oskar_telescope.h"
#include "utility/oskar_file_exists.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"
#include "vis/oskar_vis_block.h"
#include "vis/oskar_vis_header.h"

#include <iostream>
#include <string>
#include <vector>

using namespace oskar;
using namespace std;

static const char app[] = "oskar_vis_add_noise";
static const char app_s[] = "oskar_sim_interferometer";


int main(int argc, char** argv)
{
    int status = 0;

    // Obtain command line options & arguments.
    OptionParser opt(app, oskar_version_string(), oskar_app_settings(app_s));
    opt.set_description("Application to add noise to OSKAR binary visibility "
            "files.");
    opt.add_required("OSKAR visibility file(s)...");
    opt.add_flag("-s", "OSKAR settings file (noise settings).", 1, "", true);
    opt.add_flag("-v", "Verbose logging.");
    opt.add_flag("-q", "Suppress all logging output.");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;

    string settings;
    opt.get("-s")->getString(settings);
    vector<string> vis_filename_in = opt.get_input_files();
    int num_files = (int)vis_filename_in.size();
    bool verbose = opt.is_set("-v") ? true : false;
    bool quiet   = opt.is_set("-q") ? true : false;

    // Create the log.
    int file_priority = OSKAR_LOG_MESSAGE;
    int term_priority = OSKAR_LOG_STATUS;
    if (quiet) term_priority = OSKAR_LOG_WARNING;
    if (verbose) term_priority = OSKAR_LOG_DEBUG;
    oskar_Log* log = oskar_log_create(file_priority, term_priority);
    oskar_log_set_keep_file(log, false);
    oskar_log_message(log, 'M', 0, "Running binary %s", argv[0]);
    oskar_log_section(log, 'M', "Loading settings file '%s'",
            settings.c_str());

    // Load the settings file.
    SettingsTree* s = oskar_app_settings_tree(app_s, settings.c_str());
    if (!s)
    {
        oskar_log_error(log, "Failed to read settings file.");
        if (log) oskar_log_free(log);
        return EXIT_FAILURE;
    }

    // Write settings to log.
    oskar_settings_log(s, log);
    if (!s->to_int("interferometer/noise/enable", &status))
    {
        oskar_log_error(log, "Noise addition disabled in the settings.");
        oskar_log_free(log);
        SettingsTree::free(s);
        return EXIT_FAILURE;
    }

    // Set up the telescope model.
    oskar_Telescope* tel = oskar_settings_to_telescope(s, log, &status);
    oskar_telescope_analyse(tel, &status);

    // Create list of output vis file names.
    vector<string> vis_filename_out(num_files);
    for (int i = 0; i < (int)vis_filename_out.size(); ++i)
    {
        if (status) break;
        string str = vis_filename_in[i];
        if (!oskar_file_exists(str.c_str()))
        {
            oskar_log_error(log, "Visibility file %s not found.", str.c_str());
            status = OSKAR_ERR_FILE_IO;
            break;
        }
        // TODO check if the file exists
        str.erase(str.end()-4, str.end());
        vis_filename_out[i] = str + "_noise.vis";
    }

    // Print a summary of what is about to happen.
    if (!status)
    {
        oskar_log_line(log, 'D', ' ');
        oskar_log_line(log, 'D', '-');
        oskar_log_value(log, 'D', -1, "Number of input files", "%i", num_files);
        for (int i = 0; i < num_files; ++i)
            oskar_log_message(log, 'D', 1, "%s", vis_filename_in[i].c_str());
        oskar_log_value(log, 'D', -1, "Settings file", "%s",
                settings.c_str());
        oskar_log_value(log, 'D', -1, "Verbose", "%s",
                verbose ? "true" : "false");
        oskar_log_line(log, 'D', '-');
    }

    // Add uncorrelated noise to each of the visibility files.
    for (int i = 0; i < num_files; ++i)
    {
        if (status) break;
        const char* in_file = vis_filename_in[i].c_str();
        const char* out_file = vis_filename_out[i].c_str();
        oskar_log_line(log, 'D', ' ');
        oskar_log_value(log, 'D', -1, "Loading visibility file", "%s", in_file);

        // Load the input file and create the output file.
        oskar_Binary* h_in = oskar_binary_create(in_file, 'r', &status);
        oskar_VisHeader* hdr = oskar_vis_header_read(h_in, &status);
        oskar_Binary* h_out = oskar_vis_header_write(hdr, out_file, &status);

        // TODO Check that the visibility file was written with normalise
        // beam mode enabled. If not print a warning.
        // TODO Also verify any settings in the vis file against those loaded.

        // Get header data.
        int max_times_per_block = oskar_vis_header_max_times_per_block(hdr);
        int num_times = oskar_vis_header_num_times_total(hdr);
        int num_blocks = (num_times + max_times_per_block - 1) /
                max_times_per_block;
        int type = oskar_vis_header_coord_precision(hdr);
        oskar_Mem* station_work = oskar_mem_create(type, OSKAR_CPU, 0, &status);

        // Create a visibility block to read into.
        oskar_VisBlock* blk = oskar_vis_block_create_from_header(OSKAR_CPU,
                hdr, &status);

        // Loop over blocks and add noise to each one.
        for (int b = 0; b < num_blocks; ++b)
        {
            if (status) break;
            oskar_vis_block_read(blk, hdr, h_in, b, &status);
            oskar_vis_block_add_system_noise(blk, hdr, tel, b, station_work,
                    &status);
            oskar_vis_block_write(blk, h_out, b, &status);
        }

        // Free memory for vis header and vis block, and close files.
        oskar_mem_free(station_work, &status);
        oskar_vis_block_free(blk, &status);
        oskar_vis_header_free(hdr, &status);
        oskar_binary_free(h_in);
        oskar_binary_free(h_out);
    }

    // Free telescope model.
    oskar_telescope_free(tel, &status);
    if (status)
        oskar_log_error(log, "Error: %s", oskar_get_error_string(status));
    oskar_log_free(log);
    SettingsTree::free(s);
    return status == 0 ? 0 : EXIT_FAILURE;
}
