/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "log/oskar_log.h"
#include "settings/oskar_option_parser.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_version_string.h"

#include <cstdio>
#include <cstdlib>
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
    double min_peak_fraction = opt.get_double("-f");
    double min_abs_val = opt.get_double("-n");
    double spectral_index = opt.get_double("-s");

    // Load the FITS image data.
    oskar_Sky* sky = oskar_sky_from_fits_file(OSKAR_DOUBLE, opt.get_arg(0),
            min_peak_fraction, min_abs_val, "Jy/beam", 0, 0.0, spectral_index,
            &error);

    // Write out the sky model.
    oskar_sky_save(sky, opt.get_arg(1), &error);
    if (error)
    {
        oskar_log_error(0, oskar_get_error_string(error));
    }

    oskar_sky_free(sky, &error);
    return error ? EXIT_FAILURE : EXIT_SUCCESS;
}
