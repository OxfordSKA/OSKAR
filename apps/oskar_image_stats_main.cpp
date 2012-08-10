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

#include "oskar_global.h"
#include "imaging/oskar_Image.h"
#include "imaging/oskar_ImageStats.h"
#include "imaging/oskar_image_get_stats.h"
#include "imaging/oskar_image_read.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_log_error.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_Log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

void usage(bool verbose = false);

int main(int argc, char** argv)
{
    int error = OSKAR_SUCCESS;

    // Parse the command line
    if (argc < 2)
    {
        usage();
        return OSKAR_ERR_INVALID_ARGUMENT;
    }

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-h") == 0)
        {
            usage(true);
            return OSKAR_SUCCESS;
        }
    }

    // Retrieve the command arguments.
    const char* filename = argv[1];
    int p = 0, t = 0, c = 0;
    bool exp_format = false;
    if (argc == 6)
    {
        if ((strcmp(argv[5], "e") == 0))
            exp_format = true;
        else if ((strcmp(argv[5], "f") == 0))
            exp_format = false;
        else
        {
            fprintf(stderr, "ERROR: Unrecognised print format specified "
                    "(argument 5), allowed values are 'f' or 'e'\n");
            return OSKAR_ERR_INVALID_ARGUMENT;
        }
        c = atoi(argv[4]);
        t = atoi(argv[3]);
        p = atoi(argv[2]);
    }
    else if (argc == 5)
    {
        c = atoi(argv[4]);
        t = atoi(argv[3]);
        p = atoi(argv[2]);
    }
    else if (argc == 4)
    {
        t = atoi(argv[3]);
        p = atoi(argv[2]);
    }
    else if (argc == 3)
    {
        p = atoi(argv[2]);
    }

    // Create the log.
    try
    {
        // Load the image into memory.
        oskar_Image image;
        error = oskar_image_read(&image, filename, 0);
        if (error)
        {
            fprintf(stderr, "ERROR: Failed to open specified image file: %s.\n",
                    oskar_get_error_string(error));
            return error;
        }

        // Calculate the image stats
        oskar_ImageStats stats;
        oskar_image_get_stats(&stats, &image, p, t, c, &error);
        if (error)
        {
            fprintf(stderr, "ERROR: Failed evaluate image statistics: %s.\n",
                    oskar_get_error_string(error));
            if (OSKAR_ERR_DIMENSION_MISMATCH)
            {
                fprintf(stderr, "  Check specified image indices are valid.\n");
            }
            return error;
        }
        if (exp_format)
            printf("%e, %e, %e, %e, %e, %e\n", stats.min, stats.max, stats.mean,
                    stats.rms, stats.var, stats.std);
        else
            printf("%f, %f, %f, %f, %f, %f\n", stats.min, stats.max, stats.mean,
                    stats.rms, stats.var, stats.std);

    }
    catch (int code)
    {
        error = code;
    }

    // Check for errors.
    if (error)
    {
        fprintf(stderr, "ERROR: Run failed with code %i: %s.\n", error,
                oskar_get_error_string(error));
    }

    return error;
}

void usage(bool verbose)
{
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  oskar_image_stats [OSKAR image] <polarisation> <time> <channel> <print format>\n");
    fprintf(stderr, "\n");
    if (verbose)
    {
        fprintf(stderr, "Arguments:\n");
        fprintf(stderr, "  OSKAR image (required): OSKAR image file path:\n");
        fprintf(stderr, "  polarisation (optional, default=0): index into the image cube\n");
        fprintf(stderr, "  time (optional, default=0): index into the image cube\n");
        fprintf(stderr, "  channel (optional, default=0): index into the image cube\n");
        fprintf(stderr, "  print format (optional, default='f'): print format for output string, 'e' or 'f'\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Output:\n");
        fprintf(stderr, "  A comma separated list of image statistics in the following format:\n");
        fprintf(stderr, "    min, max, mean, rms, variance, std.dev.\n" );
        fprintf(stderr, "\n");
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "  $ oskar_image_stats my_image.img 1 2 3 f\n");
    }
    else
    {
        fprintf(stderr, "  For more detailed usage information run with '-h'\n");
    }
    fprintf(stderr, "\n");
}


