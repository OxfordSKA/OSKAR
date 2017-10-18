/*
 * Copyright (c) 2015, The University of Oxford
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
#include "binary/oskar_binary.h"
#include "log/oskar_log.h"
#include "mem/oskar_binary_read_mem.h"
#include "utility/oskar_version_string.h"
#include "utility/oskar_get_error_string.h"
#include "vis/oskar_vis.h"

#include <string>
#include <cstdio>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace std;

int main(int argc, char** argv)
{
    int error = 0;

    oskar::OptionParser opt("oskar_vis_upgrade_format", oskar_version_string());
    opt.set_description("Upgrades one or more old OSKAR visibility binary "
            "files to the current format version.");
    opt.add_required("OSKAR visibility file(s)");
    opt.add_example("oskar_vis_upgrade_format file1.vis");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;

    // Get the options.
    string out_path;
    vector<string> in_files = opt.get_input_files(1);
    bool verbose = opt.is_set("-q") ? false : true;
    int num_in_files = (int)in_files.size();

    // Print if verbose.
    if (verbose)
    {
        cout << "Using the " << num_in_files << " input files:" << endl;
        for (int i = 0; i < num_in_files; ++i)
        {
            cout << "  [" << setw(2) << i << "] " << in_files[i] << endl;
        }
    }

    // Loop over input files.
    for (int i = 0; i < num_in_files; ++i)
    {
        std::string out;

        // Break on error.
        if (error) break;

        // Get the name of the current input file.
        const char* in_file = in_files[i].c_str();
        out = in_files[i] + ".upgraded.vis";

        // Load the visibility file.
        oskar_Binary* h = oskar_binary_create(in_file, 'r', &error);
        oskar_Vis* vis = oskar_vis_read(h, &error);
        oskar_binary_free(h);

        // Write data in new format and free the structure.
        oskar_vis_write(vis, 0, out.c_str(), &error);
        oskar_vis_free(vis, &error);
    }

    if (error)
        oskar_log_error(0, oskar_get_error_string(error));
    return error;
}
