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

#include "sky/oskar_rebin_sky_cuda.h"
#include "sky/oskar_SkyModel.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_log_error.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    int error = OSKAR_SUCCESS;

    // Parse command line.
    if (argc != 2)
    {
        fprintf(stderr, "Usage: $ oskar_rebin_sky [input sky file] "
                "[output sky file]\n");
        return OSKAR_ERR_INVALID_ARGUMENT;
    }

    // Load input and output sky models.
    oskar_SkyModel input, output;
    oskar_sky_model_load(&input, argv[1], &error);
    if (error)
    {
        fprintf(stderr, "Error loading input sky file.\n");
        return OSKAR_ERR_FILE_IO;
    }
    oskar_sky_model_load(&output, argv[2], &error);
    if (error)
    {
        fprintf(stderr, "Error loading output sky file.\n");
        return OSKAR_ERR_FILE_IO;
    }

    // Rebin flux in input sky to output source positions.

    return error;
}
