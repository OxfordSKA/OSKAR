/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "oskar_rebin_sky_cuda.h"
#include "log/oskar_log.h"
#include "settings/oskar_option_parser.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_error_string.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    oskar_Sky *input, *output, *input_gpu, *output_gpu;
    int error = 0;

    oskar::OptionParser opt("oskar_rebin_sky");
    opt.add_required("input sky file");
    opt.add_required("output sky file");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;

    // Load input and output sky models.
    printf("Loading input '%s'\n", argv[1]);
    input = oskar_sky_load(argv[1], OSKAR_SINGLE, &error);
    if (error)
    {
        fprintf(stderr, "Error loading input sky file.\n");
        return EXIT_FAILURE;
    }
    printf("Loading output '%s'\n", argv[2]);
    output = oskar_sky_load(argv[2], OSKAR_SINGLE, &error);
    if (error)
    {
        fprintf(stderr, "Error loading output sky file.\n");
        return EXIT_FAILURE;
    }

    // Copy sky models to GPU.
    input_gpu = oskar_sky_create_copy(input, OSKAR_GPU, &error);
    output_gpu = oskar_sky_create_copy(output, OSKAR_GPU, &error);

    // Free CPU sky models.
    oskar_sky_free(input, &error);
    oskar_sky_free(output, &error);

    // Rebin flux in input sky to output source positions.
    oskar_mem_clear_contents(oskar_sky_I(output_gpu), &error);
    oskar_rebin_sky_cuda_f(
            oskar_sky_num_sources(input_gpu),
            oskar_sky_num_sources(output_gpu),
            oskar_mem_float_const(oskar_sky_ra_rad_const(input_gpu), &error),
            oskar_mem_float_const(oskar_sky_dec_rad_const(input_gpu), &error),
            oskar_mem_float_const(oskar_sky_I_const(input_gpu), &error),
            oskar_mem_float_const(oskar_sky_ra_rad_const(output_gpu), &error),
            oskar_mem_float_const(oskar_sky_dec_rad_const(output_gpu), &error),
            oskar_mem_float(oskar_sky_I(output_gpu), &error));
    oskar_device_check_error_cuda(&error);
    if (error)
        fprintf(stderr, "CUDA error (%s).\n", oskar_get_error_string(error));

    // Write new sky model out.
    output = oskar_sky_create_copy(output_gpu, OSKAR_CPU, &error);
    oskar_sky_save(output, argv[2], &error);

    // Free sky models.
    oskar_sky_free(input_gpu, &error);
    oskar_sky_free(output_gpu, &error);
    oskar_sky_free(output, &error);

    return error ? EXIT_FAILURE : EXIT_SUCCESS;
}
