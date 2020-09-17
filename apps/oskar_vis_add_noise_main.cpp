/*
 * Copyright (c) 2013-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "apps/oskar_app_settings.h"
#include "apps/oskar_settings_log.h"
#include "apps/oskar_settings_to_telescope.h"
#include "binary/oskar_binary.h"
#include "log/oskar_log.h"
#include "settings/oskar_option_parser.h"
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

    int num_files = 0;
    string settings(opt.get_string("-s"));
    const char* const* vis_filename_in = opt.get_input_files(1, &num_files);
    bool verbose = opt.is_set("-v") ? true : false;
    bool quiet   = opt.is_set("-q") ? true : false;

    // Set log parameters.
    oskar_Log* log = 0;
    int file_priority = OSKAR_LOG_MESSAGE;
    int term_priority = OSKAR_LOG_STATUS;
    if (quiet) term_priority = OSKAR_LOG_WARNING;
    if (verbose) term_priority = OSKAR_LOG_DEBUG;
    oskar_log_set_file_priority(log, file_priority);
    oskar_log_set_term_priority(log, term_priority);
    oskar_log_set_keep_file(log, false);

    // Load the settings file.
    SettingsTree* s = oskar_app_settings_tree(app_s, settings.c_str());
    if (!s)
    {
        oskar_log_error(0, "Failed to read settings file.");
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
    oskar_Telescope* tel = oskar_settings_to_telescope(s, 0, &status);
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
            oskar_log_message(log, 'D', 1, "%s", vis_filename_in[i]);
        oskar_log_value(log, 'D', -1, "Settings file", "%s", settings.c_str());
        oskar_log_value(log, 'D', -1,
                "Verbose", "%s", verbose ? "true" : "false");
        oskar_log_line(log, 'D', '-');
    }

    // Add uncorrelated noise to each of the visibility files.
    for (int i = 0; i < num_files; ++i)
    {
        if (status) break;
        const char* in_file = vis_filename_in[i];
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
        int type = oskar_vis_header_coord_precision(hdr);
        oskar_Mem* station_work = oskar_mem_create(type, OSKAR_CPU, 0, &status);

        // Create a visibility block to read into.
        oskar_VisBlock* blk = oskar_vis_block_create_from_header(OSKAR_CPU,
                hdr, &status);

        // Loop over blocks and add noise to each one.
        const int num_blocks = oskar_vis_header_num_blocks(hdr);
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
