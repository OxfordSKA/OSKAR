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

#include <apps/lib/oskar_vis_block_write_ms.h>
#include <apps/lib/oskar_vis_header_write_ms.h>
#include <apps/lib/oskar_OptionParser.h>
#include <oskar_get_error_string.h>
#include <oskar_log.h>
#include <oskar_vis_header.h>
#include <oskar_vis_block.h>
#include <oskar_measurement_set.h>
#include <oskar_version_string.h>
#include <oskar_binary.h>
#include <oskar_binary_read_mem.h>
#include <cstdio>
#include <string>
#include <vector>

using namespace std;

// Check if built with Measurement Set support.
#ifndef OSKAR_NO_MS
int main(int argc, char** argv)
{
    int error = 0;

    oskar_OptionParser opt("oskar_vis_to_ms", oskar_version_string());
    opt.setDescription("Converts one or more OSKAR visibility binary files to "
            "Measurement Set format (http://casa.nrao.edu/Memos/229.html).\n"
            "If not specified, the name of the Measurement Set will "
            "be 'out.ms'.");
    opt.addRequired("OSKAR visibility files...");
    opt.addFlag("-o", "Output Measurement Set name", 1, "out.ms",
            false, "--output");
    opt.addFlag("-p", "Force polarised MS format", false, "--force_polarised");
    opt.addExample("oskar_vis_to_ms file1.vis file2.vis");
    opt.addExample("oskar_vis_to_ms file1.vis file2.vis -o stitched.ms");
    opt.addExample("oskar_vis_to_ms *.vis");
    if (!opt.check_options(argc, argv))
        return OSKAR_FAIL;

    // Get the options.
    string out_path;
    opt.get("-o")->getString(out_path);
    vector<string> in_files = opt.getInputFiles(1);
    bool verbose = opt.isSet("-q") ? false : true;
    bool force_polarised = opt.isSet("-p") ? true : false;
    int num_in_files = in_files.size();

    // Print if verbose.
    if (verbose)
    {
        printf("Output Measurement Set: %s\n", out_path.c_str());
        printf("Using the %d input files:\n", num_in_files);
        for (int i = 0; i < num_in_files; ++i)
        {
            printf("  [%02d] %s\n", i, in_files[i].c_str());
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
        const char* in_file = in_files[i].c_str();
        oskar_Binary* h = oskar_binary_create(in_file, 'r', &error);
        if (error) break;
        oskar_VisHeader* hdr = oskar_vis_header_read(h, &error);
        if (error) break;

        // Create the Measurement Set using the header from the first file.
        if (i == 0)
        {
            ms = oskar_vis_header_write_ms(hdr, out_path.c_str(), 1,
                    force_polarised, &error);
        }

        // Work out the expected number of blocks in the file.
        int max_times_per_block = oskar_vis_header_max_times_per_block(hdr);
        int num_times = oskar_vis_header_num_times_total(hdr);
        int num_blocks = (num_times + max_times_per_block - 1) /
                max_times_per_block;

        // Create a visibility block to read into.
        oskar_VisBlock* blk = oskar_vis_block_create(OSKAR_CPU, hdr, &error);

        // Loop over blocks and write them to the Measurement Set.
        for (int b = 0; b < num_blocks; ++b)
        {
            oskar_vis_block_read(blk, hdr, h, b, &error);
            oskar_vis_block_write_ms(blk, hdr, ms, &error);
        }

        // Add run log to Measurement Set.
        oskar_Mem* log = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, &error);
        oskar_binary_read_mem(h, log, OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0,
                &tag_error);
        oskar_ms_add_log(ms, oskar_mem_char_const(log), oskar_mem_length(log));

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
    oskar_log_error(0, "OSKAR was not compiled with Measurement Set support.");
    return OSKAR_FAIL;
}
#endif


