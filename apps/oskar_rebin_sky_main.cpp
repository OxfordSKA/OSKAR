/*
 * Copyright (c) 2012, The University of Oxford
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

#include <cuda_runtime_api.h>
#include "sky/oskar_rebin_sky_cuda.h"
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_sky_model_copy.h"
#include "sky/oskar_sky_model_free.h"
#include "sky/oskar_sky_model_init.h"
#include "sky/oskar_sky_model_load.h"
#include "sky/oskar_sky_model_save.h"
#include "utility/oskar_cuda_check_error.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_clear_contents.h"
#include "utility/oskar_log_error.h"

#include <apps/lib/oskar_OptionParser.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    oskar_SkyModel input, output, input_gpu, output_gpu;
    int error = 0;

    oskar_OptionParser opt("oskar_rebin_sky");
    opt.addRequired("input sky file");
    opt.addRequired("output sky file");
    if (!opt.check_options(argc, argv))
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Initialise sky models.
    oskar_sky_model_init(&input, OSKAR_SINGLE, OSKAR_LOCATION_CPU, 0, &error);
    oskar_sky_model_init(&output, OSKAR_SINGLE, OSKAR_LOCATION_CPU, 0, &error);
    oskar_sky_model_init(&input_gpu, OSKAR_SINGLE, OSKAR_LOCATION_GPU, 0, &error);
    oskar_sky_model_init(&output_gpu, OSKAR_SINGLE, OSKAR_LOCATION_GPU, 0, &error);

    // Load input and output sky models.
    printf("Loading input '%s'\n", argv[1]);
    oskar_sky_model_load(&input, argv[1], &error);
    if (error)
    {
        fprintf(stderr, "Error loading input sky file.\n");
        return OSKAR_ERR_FILE_IO;
    }
    printf("Loading output '%s'\n", argv[2]);
    oskar_sky_model_load(&output, argv[2], &error);
    if (error)
    {
        fprintf(stderr, "Error loading output sky file.\n");
        return OSKAR_ERR_FILE_IO;
    }

    // Copy sky models to GPU.
    oskar_sky_model_copy(&input_gpu, &input, &error);
    oskar_sky_model_copy(&output_gpu, &output, &error);

    // Rebin flux in input sky to output source positions.
    oskar_mem_clear_contents(&output_gpu.I, &error);
    oskar_rebin_sky_cuda_f(input_gpu.num_sources, output_gpu.num_sources,
            (const float*)(input_gpu.RA), (const float*)(input_gpu.Dec),
            (const float*)(input_gpu.I), (const float*)(output_gpu.RA),
            (const float*)(output_gpu.Dec), (float*)(output_gpu.I));
    oskar_cuda_check_error(&error);
    if (error)
        fprintf(stderr, "CUDA error (%s).\n", oskar_get_error_string(error));

    // Write new sky model out.
    oskar_sky_model_copy(&output, &output_gpu, &error);
    oskar_sky_model_save(argv[2], &output, &error);

    // Free sky models.
    oskar_sky_model_free(&input, &error);
    oskar_sky_model_free(&output, &error);
    oskar_sky_model_free(&input_gpu, &error);
    oskar_sky_model_free(&output_gpu, &error);

    return error;
}
