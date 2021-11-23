/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "log/oskar_log.h"
#include "settings/oskar_option_parser.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"
#include "vis/oskar_vis_block.h"
#include "vis/oskar_vis_header.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace std;

static bool is_compatible(
        const oskar_VisHeader* vis1, const oskar_VisHeader* vis2);

int main(int argc, char** argv)
{
    int status = 0;

    oskar::OptionParser opt("oskar_vis_add", oskar_version_string());
    opt.set_description("Application to combine OSKAR visibility files.");
    opt.add_required("OSKAR visibility files...");
    opt.add_flag("-o", "Output visibility file name", 1, "out.vis", false, "--output");
    opt.add_flag("-q", "Disable log messages", false, "--quiet");
    opt.add_example("oskar_vis_add file1.vis file2.vis");
    opt.add_example("oskar_vis_add file1.vis file2.vis -o combined.vis");
    opt.add_example("oskar_vis_add -q file1.vis file2.vis file3.vis");
    opt.add_example("oskar_vis_add *.vis");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;

    // Get the options.
    int num_in_files = 0;
    const char* const* in_files = opt.get_input_files(2, &num_in_files);
    const char* out_path = opt.get_string("-o");
    const bool verbose = opt.is_set("-q") ? false : true;
    if (num_in_files < 2)
    {
        opt.error("Please provide 2 or more visibility files to combine.");
        return EXIT_FAILURE;
    }

    // Print if verbose.
    if (verbose)
    {
        printf("Output visibility file: %s\n", out_path);
        printf("Combining the %d input files:\n", num_in_files);
        for (int i = 0; i < num_in_files; ++i)
        {
            printf("  [%02d] %s\n", i, in_files[i]);
        }
    }

    // Read all the visibility headers and check consistency.
    oskar_Binary **files = 0, *out_file = 0;
    oskar_VisHeader** headers = 0;
    oskar_VisBlock *block0 = 0, *block1 = 0;
    files = (oskar_Binary**) calloc(num_in_files, sizeof(oskar_Binary*));
    headers = (oskar_VisHeader**) calloc(num_in_files, sizeof(oskar_VisHeader*));
    for (int i = 0; i < num_in_files; ++i)
    {
        files[i] = oskar_binary_create(in_files[i], 'r', &status);
        headers[i] = oskar_vis_header_read(files[i], &status);
        if (status)
        {
            oskar_log_error(0,
                    "Failed to read visibility data file '%s'", in_files[i]);
            break;
        }
        if (i > 0 && !is_compatible(headers[0], headers[i]))
        {
            status = OSKAR_ERR_TYPE_MISMATCH;
            oskar_log_error(0,
                    "Input visibility file '%s' does not have "
                    "the same dimensions", in_files[i]);
            break;
        }
    }

    // Write the output file using the first header.
    if (!status)
    {
        block0 = oskar_vis_block_create_from_header(
                OSKAR_CPU, headers[0], &status);
        block1 = oskar_vis_block_create_from_header(
                OSKAR_CPU, headers[0], &status);
        oskar_mem_clear_contents(oskar_vis_header_settings(
                headers[0]), &status);
        oskar_mem_clear_contents(oskar_vis_header_telescope_path(
                headers[0]), &status);
        out_file = oskar_vis_header_write(headers[0], out_path, &status);
        if (status)
        {
            oskar_log_error(0,
                    "Failed to write output visibility data file '%s'",
                    out_path);
        }
    }

    // Loop over each block in each file.
    const int num_blocks = oskar_vis_header_num_blocks(headers[0]);
    for (int b = 0; (b < num_blocks) && !status; ++b)
    {
        // Read reference block.
        oskar_vis_block_read(block0, headers[0], files[0], b, &status);

        // Read blocks from other files and combine them.
        for (int i = 1; (i < num_in_files) && !status; ++i)
        {
            oskar_vis_block_read(block1, headers[i], files[i], b, &status);
            if (status)
            {
                oskar_log_error(0, "Failed to read visibility block in '%s'",
                        in_files[i]);
                break;
            }
            oskar_Mem* b0 = oskar_vis_block_cross_correlations(block0);
            const oskar_Mem* b1 =
                    oskar_vis_block_cross_correlations_const(block1);
            oskar_mem_add(b0, b0, b1, 0, 0, 0, oskar_mem_length(b0), &status);
        }

        // Write combined block.
        oskar_vis_block_write(block0, out_file, b, &status);
    }

    // Free memory and close files.
    oskar_vis_block_free(block0, &status);
    oskar_vis_block_free(block1, &status);
    for (int i = 0; i < num_in_files; ++i)
    {
        oskar_vis_header_free(headers[i], &status);
        oskar_binary_free(files[i]);
    }
    oskar_binary_free(out_file);
    free(files);
    free(headers);

    return status ? EXIT_FAILURE : EXIT_SUCCESS;
}

static bool is_compatible(const oskar_VisHeader* v1, const oskar_VisHeader* v2)
{
    if (oskar_vis_header_num_channels_total(v1) !=
            oskar_vis_header_num_channels_total(v2))
    {
        return false;
    }
    if (oskar_vis_header_max_channels_per_block(v1) !=
            oskar_vis_header_max_channels_per_block(v2))
    {
        return false;
    }
    if (oskar_vis_header_num_times_total(v1) !=
            oskar_vis_header_num_times_total(v2))
    {
        return false;
    }
    if (oskar_vis_header_max_times_per_block(v1) !=
            oskar_vis_header_max_times_per_block(v2))
    {
        return false;
    }
    if (oskar_vis_header_num_stations(v1) !=
            oskar_vis_header_num_stations(v2))
    {
        return false;
    }
    if (fabs(oskar_vis_header_freq_start_hz(v1) -
            oskar_vis_header_freq_start_hz(v2)) > DBL_EPSILON)
    {
        return false;
    }
    if (fabs(oskar_vis_header_freq_inc_hz(v1) -
            oskar_vis_header_freq_inc_hz(v2)) > DBL_EPSILON)
    {
        return false;
    }
    if (oskar_vis_header_amp_type(v1) != oskar_vis_header_amp_type(v2))
    {
        return false;
    }
    if (oskar_vis_header_pol_type(v1) != oskar_vis_header_pol_type(v2))
    {
        return false;
    }

    return true;
}
