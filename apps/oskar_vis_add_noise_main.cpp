/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include <oskar_vis_block.h>
#include <oskar_vis_header.h>
#include <oskar_telescope.h>
#include <oskar_file_exists.h>

#include <oskar_binary.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

//-----------------------------------------------------------------------------
//void log_noise_settings(oskar_Log* log, oskar_Settings* settings);
//-----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    int status = 0;

    // Obtain command line options & arguments.
    oskar_OptionParser opt("oskar_vis_add_noise", oskar_version_string());
    opt.setDescription("Application to add noise to OSKAR binary visibility "
            "files.");
    opt.addRequired("OSKAR visibility file(s)...");
    opt.addFlag("-s", "OSKAR settings file (noise settings).", 1, "", true);
    opt.addFlag("-v", "Verbose logging.");
    opt.addFlag("-q", "Suppress all logging output.");
    if (!opt.check_options(argc, argv))
        return EXIT_FAILURE;

    string settings_file;
    opt.get("-s")->getString(settings_file);
    vector<string> vis_filename_in = opt.getInputFiles();
    int num_files = vis_filename_in.size();
    bool verbose = opt.isSet("-v") ? true : false;
    bool quiet   = opt.isSet("-q") ? true : false;

    // Create the log.
    int file_priority = OSKAR_LOG_MESSAGE;
    int term_priority = OSKAR_LOG_STATUS;
    if (quiet) term_priority = OSKAR_LOG_WARNING;
    if (verbose) term_priority = OSKAR_LOG_DEBUG;
    oskar_Log* log = oskar_log_create(file_priority, term_priority);
    oskar_log_set_keep_file(log, false);
    oskar_log_message(log, 'M', 0, "Running binary %s", argv[0]);

    // Load the settings file and telescope model.
    oskar_log_section(log, 'M', "Loading settings file '%s'",
            settings_file.c_str());
    oskar_Settings settings;
    oskar_settings_load(&settings, 0, settings_file.c_str(), &status);
    if (status != OSKAR_SUCCESS) {
        oskar_log_error(log, "Failed to load settings from '%s'",
                settings_file.c_str());
        oskar_log_free(log);
        return EXIT_FAILURE;
    }
    if (!settings.interferometer.noise.enable)
    {
        oskar_log_error(log, "Noise addition disabled in the settings.");
        oskar_log_free(log);
        return EXIT_FAILURE;
    }

    // FIXME oskar_set_up_telescope should not be printing log messages!
    oskar_Telescope* tel = oskar_set_up_telescope(&settings, log, &status);
    if (status)
    {
        oskar_log_error(log, "Error: %s", oskar_get_error_string(status));
        oskar_telescope_free(tel, &status);
        oskar_log_free(log);
        return EXIT_FAILURE;
    }

    // Create list of output vis file names.
    vector<string> vis_filename_out(num_files);
    for (int i = 0; i < (int)vis_filename_out.size(); ++i)
    {
        string str = vis_filename_in[i];
        if (!oskar_file_exists(str.c_str())) {
            oskar_log_error(log, "Visibility file %s not found.", str.c_str());
            oskar_log_free(log);
            return EXIT_FAILURE;
        }
        // TODO check if the file exists
        str.erase(str.end()-4, str.end());
        vis_filename_out[i] = str + "_noise.vis";
    }

    // Print a summary of what is about to happen.
    oskar_log_line(log, 'D', ' ');
    oskar_log_line(log, 'D', '-');
    oskar_log_value(log, 'D', -1, "Number of input files", "%li", num_files);
    for (int i = 0; i < num_files; ++i)
        oskar_log_message(log, 'D', 1, "%s", vis_filename_in[i].c_str());
    oskar_log_value(log, 'D', -1, "Settings file", "%s", settings_file.c_str());
    oskar_log_value(log, 'D', -1, "Verbose", "%s", (verbose?"true":"false"));
    oskar_log_line(log, 'D', '-');

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
        oskar_VisBlock* blk = oskar_vis_block_create(OSKAR_CPU, hdr, &status);

        // Loop over blocks.
        for (int b = 0; b < num_blocks; ++b)
        {
            // Check for errors.
            if (status) break;

            // Read the block.
            oskar_vis_block_read(blk, hdr, h_in, b, &status);

            // Add noise to the block.
            oskar_vis_block_add_system_noise(blk, hdr, tel,
                    settings.interferometer.noise.seed, b, station_work,
                    &status);

            // Write the block.
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
    return status;
}
