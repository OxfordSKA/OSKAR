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
#ifndef OSKAR_NO_FITS
#include "fits/oskar_fits_to_sky_model.h"
#endif
#include "sky/oskar_sky_model_write.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_log_error.h"
#include <cstdio>

int main(int argc, char** argv)
{
    // Check if built with FITS support.
#ifndef OSKAR_NO_FITS
    int error;

    // Parse command line.
    double spectral_index = 0.0;
    double min_peak_fraction = 0.0;
    double noise_floor = 0.0;
    int downsample_factor = 0;
    if (argc < 3)
    {
        fprintf(stderr,
                "Usage: $ oskar_fits_to_sky_model [FITS file] [sky model file]\n"
                "    [spectral index] [min peak fraction]\n"
                "    [noise floor] [downsample factor]\n");
        return OSKAR_ERR_INVALID_ARGUMENT;
    }

    if (argc > 3)
        spectral_index = strtod(argv[3], 0);
    if (argc > 4)
        min_peak_fraction = strtod(argv[4], 0);
    if (argc > 5)
        noise_floor = strtod(argv[5], 0);
    if (argc > 6)
        downsample_factor = (int)strtol(argv[6], 0, 0);

    // Load the FITS file as a sky model.
    oskar_SkyModel sky;
    error = oskar_fits_to_sky_model(0, argv[1], &sky, spectral_index,
            min_peak_fraction, noise_floor, downsample_factor);
    if (error)
    {
        oskar_log_error(0, oskar_get_error_string(error));
        return error;
    }

    // Write out the sky model.
    oskar_sky_model_write(argv[2], &sky, &error);
    if (error)
    {
        oskar_log_error(0, oskar_get_error_string(error));
        return error;
    }
    return OSKAR_SUCCESS;

#else
    // No FITS support.
    oskar_log_error(0, "OSKAR was not compiled with FITS support.");
    return -1;
#endif
}
