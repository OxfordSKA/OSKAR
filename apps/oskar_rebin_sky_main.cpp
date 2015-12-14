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

#include <oskar_rebin_sky_cuda.h>
#include <oskar_sky.h>
#include <oskar_cuda_check_error.h>
#include <oskar_get_error_string.h>
#include <oskar_log.h>

#include <apps/lib/oskar_OptionParser.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    oskar_Sky *input, *output, *input_gpu, *output_gpu;
    int error = 0;

    oskar_OptionParser opt("oskar_rebin_sky");
    opt.addRequired("input sky file");
    opt.addRequired("output sky file");
    if (!opt.check_options(argc, argv))
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Load input and output sky models.
    printf("Loading input '%s'\n", argv[1]);
    input = oskar_sky_load(argv[1], OSKAR_SINGLE, &error);
    if (error)
    {
        fprintf(stderr, "Error loading input sky file.\n");
        return OSKAR_ERR_FILE_IO;
    }
    printf("Loading output '%s'\n", argv[2]);
    output = oskar_sky_load(argv[2], OSKAR_SINGLE, &error);
    if (error)
    {
        fprintf(stderr, "Error loading output sky file.\n");
        return OSKAR_ERR_FILE_IO;
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
    oskar_cuda_check_error(&error);
    if (error)
        fprintf(stderr, "CUDA error (%s).\n", oskar_get_error_string(error));

    // Write new sky model out.
    output = oskar_sky_create_copy(output_gpu, OSKAR_CPU, &error);
    oskar_sky_save(argv[2], output, &error);

    // Free sky models.
    oskar_sky_free(input_gpu, &error);
    oskar_sky_free(output_gpu, &error);
    oskar_sky_free(output, &error);

    return error;
}
