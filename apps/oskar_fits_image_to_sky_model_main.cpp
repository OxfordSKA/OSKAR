/*
 * Copyright (c) 2012-2016, The University of Oxford
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
#include "log/oskar_log.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv)
{
    int error = 0;

    oskar::OptionParser opt("oskar_fits_image_to_sky_model",
            oskar_version_string());
    opt.set_description("Converts a FITS image to an OSKAR sky model. A number "
            "of options are provided to control how much of the image is used "
            "to make the sky model.");
    opt.add_required("FITS file", "The input FITS image to convert.");
    opt.add_required("sky model file", "The output OSKAR sky model file name to save.");
    opt.add_flag("-s", "Spectral index. This is the spectral index that will "
            "be given to each pixel in the output sky model.", 1, "0.0");
    opt.add_flag("-f", "Minimum allowed fraction of image peak. Pixel values "
            "below this fraction will be ignored.", 1, "0.0");
    opt.add_flag("-n", "Noise floor in units of original image. "
            "Pixels below this value will be ignored.", 1, "0.0");
    if (!opt.check_options(argc, argv)) return EXIT_FAILURE;

    // Parse command line.
    double spectral_index = 0.0;
    double min_peak_fraction = 0.0;
    double min_abs_val = 0.0;
    opt.get("-f")->getDouble(min_peak_fraction);
    opt.get("-n")->getDouble(min_abs_val);
    opt.get("-s")->getDouble(spectral_index);

    // Load the FITS image data.
    oskar_Sky* sky = oskar_sky_from_fits_file(OSKAR_DOUBLE, opt.get_arg(0),
            min_peak_fraction, min_abs_val, "Jy/beam", 0, 0.0, spectral_index,
            &error);

    // Write out the sky model.
    oskar_sky_save(opt.get_arg(1), sky, &error);
    if (error)
    {
        oskar_log_error(0, oskar_get_error_string(error));
        return error;
    }

    oskar_sky_free(sky, &error);
    return 0;
}
