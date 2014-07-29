/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <apps/lib/oskar_vis_write_ms.h>
#include <apps/lib/oskar_OptionParser.h>
#include <oskar_get_error_string.h>
#include <oskar_log.h>
#include <oskar_vis.h>
#include <oskar_version_string.h>
#include <oskar_binary.h>
#include <string>
#include <cstdio>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace std;

int main(int argc, char** argv)
{
    // Check if built with Measurement Set support.
#ifndef OSKAR_NO_MS
    int error = 0;

    oskar_OptionParser opt("oskar_vis_to_ms", oskar_version_string());
    opt.setDescription("Converts one or more OSKAR visibility binary files to "
            "Measurement Set format (http://casa.nrao.edu/Memos/229.html).\n"
            "If not specified, the name of the Measurement Set will "
            "be 'out.ms'.");
    opt.addRequired("OSKAR visibility files...");
    opt.addFlag("-o", "Output Measurement Set name", 1, "out.ms",
            false, "--output");
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
    int num_in_files = in_files.size();

    // Print if verbose.
    if (verbose)
    {
        cout << "Output Measurement Set: " << out_path << endl;
        cout << "Using the " << num_in_files << " input files:" << endl;
        for (int i = 0; i < num_in_files; ++i)
        {
            cout << "  [" << setw(2) << i << "] " << in_files[i] << endl;
        }
    }

    // Loop over input files.
    for (int i = 0; i < num_in_files; ++i)
    {
        // Break on error.
        if (error) break;

        // Get the name of the current input file.
        const char* in_file = in_files[i].c_str();

        // Load the visibility file.
        oskar_Vis* vis = oskar_vis_read(in_file, &error);

        // Load the run log.
        oskar_Binary* h = oskar_binary_create(in_file, 'r', &error);
        oskar_Mem* log = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, &error);
        oskar_binary_read_mem(h, log, OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0,
                &error);
        oskar_binary_free(h);

        // Write data to Measurement Set.
        int overwrite = (i == 0) ? 1 : 0;
        oskar_vis_write_ms(vis, out_path.c_str(), overwrite,
                oskar_mem_char_const(log), oskar_mem_length(log), &error);

        // Clean up.
        oskar_vis_free(vis, &error);
        oskar_mem_free(log, &error);
    }

    if (error)
        oskar_log_error(0, oskar_get_error_string(error));
    return error;

#else
    // No Measurement Set support.
    oskar_log_error(0, "OSKAR was not compiled with Measurement Set support.");
    return OSKAR_FAIL;
#endif
}
