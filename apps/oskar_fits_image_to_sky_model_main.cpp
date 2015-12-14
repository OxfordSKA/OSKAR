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

#include "apps/lib/oskar_OptionParser.h"
#include "fits/oskar_fits_image_to_sky_model.h"

#include <oskar_sky.h>
#include <oskar_get_error_string.h>
#include <oskar_log.h>
#include <oskar_version_string.h>

#include <cstdio>

int main(int argc, char** argv)
{
    // Check if built with FITS support.
    int error = OSKAR_SUCCESS;

    oskar_OptionParser opt("oskar_fits_image_to_sky_model",
            oskar_version_string());
    opt.setDescription("Converts a FITS image to an OSKAR sky model. A number "
            "of options are provided to control how much of the image is used "
            "to make the sky model.");
    opt.addRequired("FITS file", "The input FITS image to convert.");
    opt.addRequired("sky model file", "The output OSKAR sky model file name to save.");
    opt.addFlag("-s", "Spectral index. This is the spectral index that will "
            "be given to each pixel in the output sky model.", 1, "0.0");
    opt.addFlag("-f", "Minimum allowed fraction of image peak. Pixel values "
            "below this fraction will be ignored.", 1, "0.0");
    opt.addFlag("-d", "Downsample factor. This is an integer that must "
            "be >= 1, and is the factor by which the image is downsampled "
            "before saving the regrouped pixel values in the sky model. "
            "For example, a downsample factor of 2 would scale the image down "
            "by 50% in both dimensions before the format conversion.", 1, "1");
    opt.addFlag("-n", "Noise floor in Jy/PIXEL. Pixels below this value "
            "will be ignored.", 1, "0.0");
    if (!opt.check_options(argc, argv))
        return OSKAR_FAIL;

    // Parse command line.
    double spectral_index = 0.0;
    double min_peak_fraction = 0.0;
    double noise_floor = 0.0;
    int downsample_factor = 1;
    opt.get("-d")->getInt(downsample_factor);
    opt.get("-f")->getDouble(min_peak_fraction);
    opt.get("-n")->getDouble(noise_floor);
    opt.get("-s")->getDouble(spectral_index);

    // Load the FITS file as a sky model.
    oskar_Sky* sky = oskar_sky_create(OSKAR_DOUBLE,
            OSKAR_CPU, 0, &error);
    error = oskar_fits_image_to_sky_model(0, opt.getArg(0), sky,
            spectral_index, min_peak_fraction, noise_floor, downsample_factor);
    if (error)
    {
        oskar_log_error(0, oskar_get_error_string(error));
        return error;
    }

    // Write out the sky model.
    oskar_sky_save(opt.getArg(1), sky, &error);
    if (error)
    {
        oskar_log_error(0, oskar_get_error_string(error));
        return error;
    }

    oskar_sky_free(sky, &error);
    return OSKAR_SUCCESS;
}
