/*
 * Copyright (c) 2012-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "binary/oskar_binary.h"
#include "log/oskar_log.h"
#include "mem/oskar_binary_read_mem.h"
#include "ms/oskar_measurement_set.h"
#include "settings/oskar_option_parser.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"
#include "vis/oskar_vis_header.h"
#include "vis/oskar_vis_block.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace std;

// Check if built with Measurement Set support.
#ifndef OSKAR_NO_MS
int main(int argc, char** argv)
{
    int error = 0;

    oskar::OptionParser opt("oskar_vis_to_ms", oskar_version_string());
    opt.set_description("Converts one or more OSKAR visibility binary files to "
            "Measurement Set format (http://casa.nrao.edu/Memos/229.html).\n"
            "If not specified, the name of the Measurement Set will "
            "be 'out.ms'.");
    opt.add_required("OSKAR visibility files...");
    opt.add_flag("-o", "Output Measurement Set name", 1, "out.ms",
            false, "--output");
    opt.add_flag("-p", "Force polarised MS format", false, "--force_polarised");
    opt.add_example("oskar_vis_to_ms file1.vis file2.vis");
    opt.add_example("oskar_vis_to_ms file1.vis file2.vis -o stitched.ms");
    opt.add_example("oskar_vis_to_ms *.vis");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;

    // Get the options.
    int num_in_files = 0;
    string out_path(opt.get_string("-o"));
    const char* const* in_files = opt.get_input_files(1, &num_in_files);
    bool verbose = opt.is_set("-q") ? false : true;
    bool force_polarised = opt.is_set("-p") ? true : false;

    // Print if verbose.
    if (verbose)
    {
        printf("Output Measurement Set: %s\n", out_path.c_str());
        printf("Using the %d input files:\n", num_in_files);
        for (int i = 0; i < num_in_files; ++i)
        {
            printf("  [%02d] %s\n", i, in_files[i]);
        }
    }

    // Handle to Measurement Set.
    oskar_MeasurementSet* ms = 0;

    // Loop over input files.
    for (int i = 0; i < num_in_files; ++i)
    {
        int tag_error = 0;

        // Break on error.
        if (error) break;

        // Read the file header.
        oskar_Binary* h = oskar_binary_create(in_files[i], 'r', &error);
        if (error) break;
        oskar_VisHeader* hdr = oskar_vis_header_read(h, &error);
        if (error) break;

        // Create the Measurement Set using the header from the first file.
        if (i == 0)
        {
            ms = oskar_vis_header_write_ms(hdr, out_path.c_str(), 1,
                    force_polarised, &error);
        }

        // Create a visibility block to read into.
        oskar_VisBlock* blk = oskar_vis_block_create_from_header(OSKAR_CPU,
                hdr, &error);

        // Loop over blocks and write them to the Measurement Set.
        const int num_blocks = oskar_vis_header_num_blocks(hdr);
        for (int b = 0; b < num_blocks; ++b)
        {
            oskar_vis_block_read(blk, hdr, h, b, &error);
            oskar_vis_block_write_ms(blk, hdr, ms, &error);
        }

        // Add run log to Measurement Set.
        oskar_Mem* log = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, &error);
        oskar_binary_read_mem(h, log, OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0,
                &tag_error);
        oskar_ms_add_history(ms, "OSKAR_LOG",
                oskar_mem_char_const(log), oskar_mem_length(log));

        // Clean up.
        oskar_binary_free(h);
        oskar_mem_free(log, &error);
        oskar_vis_header_free(hdr, &error);
        oskar_vis_block_free(blk, &error);
    }

    // Close the Measurement Set.
    oskar_ms_close(ms);
    if (error)
        oskar_log_error(0, oskar_get_error_string(error));
    return error;
}
#else
// No Measurement Set support.
int main(void)
{
    oskar_log_error(0, "OSKAR was compiled without Measurement Set support.");
    return EXIT_FAILURE;
}
#endif


