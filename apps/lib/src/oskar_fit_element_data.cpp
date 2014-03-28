/*
 * Copyright (c) 2014, The University of Oxford
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

#include <apps/lib/oskar_settings_load_element_fit.h>
#include <apps/lib/oskar_fit_element_data.h>
#include <apps/lib/oskar_Dir.h>

#include <oskar_element.h>
#include <oskar_image.h>
#include <oskar_log.h>
#include <oskar_settings_init.h>
#include <oskar_settings_free.h>
#include <oskar_get_error_string.h>

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

using std::string;

static std::string construct_element_pathname(const char* output_dir,
        int port, double frequency_hz)
{
    std::ostringstream stream;
    stream << "element_pattern_fit_";
    if (port == 1)
    {
        stream << "x_";
    }
    else if (port == 2)
    {
        stream << "y_";
    }

    // Append the frequency in MHz.
    stream << std::fixed << std::setprecision(0) << frequency_hz / 1.0e6;

    // Append the file extension.
    stream << ".bin";

    oskar_Dir dir(output_dir);
    return dir.absoluteFilePath(stream.str());
}


extern "C"
void oskar_fit_element_data(const char* settings_file, oskar_Log* log,
        int* status)
{
    int port;
    string output;

    // Load the settings.
    oskar_Settings settings;
    oskar_settings_init(&settings);
    oskar_settings_load_element_fit(&settings.element_fit, settings_file, status);
    if (*status)
    {
        oskar_log_error(log, "Failed to read settings file: %s\n",
                oskar_get_error_string(*status));
        return;
    }

    // Get the main settings.
    oskar_log_settings_element_fit(log, &settings);
    const char* input_file = settings.element_fit.input_cst_file;
    const char* output_dir = settings.element_fit.output_directory;
    const char* fits_image = settings.element_fit.fits_image;
    double frequency_hz = settings.element_fit.frequency_hz;

    // Check that the input and output files have been set.
    if (!input_file || !output_dir)
    {
        *status = OSKAR_ERR_FILE_IO;
        oskar_settings_free(&settings);
        return;
    }

    // Create an element model.
    oskar_Element* element = oskar_element_create(OSKAR_DOUBLE,
            OSKAR_LOCATION_CPU, status);

    // Load the CST text file for the correct port (X=1, Y=2).
    port = settings.element_fit.polarisation_type;
    oskar_log_message(log, 0, "Loading CST element pattern: %s", input_file);
    oskar_element_load_cst(element, log, port, frequency_hz, input_file,
            &settings.element_fit, status);

    // Construct the output file name based on the settings.
    port = settings.element_fit.polarisation_type;

    if (port == 0)
    {
        output = construct_element_pathname(output_dir, 1, frequency_hz);
        oskar_element_write(element, 1, frequency_hz, output.c_str(), status);
        output = construct_element_pathname(output_dir, 2, frequency_hz);
        oskar_element_write(element, 2, frequency_hz, output.c_str(), status);
    }
    else
    {
        output = construct_element_pathname(output_dir, port, frequency_hz);
        oskar_element_write(element, port, frequency_hz, output.c_str(),
                status);
    }

    // Check if a FITS image is required.
    if (fits_image)
    {
        // Generate an image grid.
        int image_size = 512;

        oskar_Image* image = oskar_image_create(OSKAR_DOUBLE,
                OSKAR_LOCATION_CPU, status);
        oskar_image_resize(image, image_size, image_size, 1, 1, 1, status);

        /* Set element pattern meta-data. */
        oskar_image_set_type(image, OSKAR_IMAGE_TYPE_BEAM_SCALAR);
        oskar_image_set_coord_frame(image, OSKAR_IMAGE_COORD_FRAME_HORIZON);
        oskar_image_set_grid_type(image, OSKAR_IMAGE_GRID_TYPE_RECTILINEAR);
        oskar_image_set_centre(image, 0.0, 90.0);
        oskar_image_set_fov(image, 180.0, 180.0);
        oskar_image_set_freq(image, settings.element_fit.frequency_hz,
                settings.element_fit.frequency_hz);
        oskar_image_set_time(image, 0.0, 0.0);

        oskar_image_free(image, status);
    }

    // Free memory.
    oskar_element_free(element, status);
    oskar_settings_free(&settings);
}

