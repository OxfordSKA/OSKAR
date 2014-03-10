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
#include <oskar_log.h>
#include <oskar_settings_init.h>
#include <oskar_settings_free.h>
#include <oskar_get_error_string.h>

#include <string>

using std::string;

static std::string oskar_construct_element_pathname(const char* output_dir,
        int port, double frequency_hz)
{
    string filename = "spline_data_cache";
    if (port == 1)
    {
        filename += "_x";
    }
    else if (port == 2)
    {
        filename += "_y";
    }
    // FIXME Use the frequency in kHz to construct the filename.
    filename += ".bin";

    oskar_Dir dir(output_dir);
    string pathname = dir.absoluteFilePath(filename);

    return pathname;
}


extern "C"
void oskar_fit_element_data(const char* settings_file, oskar_Log* log,
        int* status)
{
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
    double frequency_hz = settings.element_fit.frequency_hz;

    // Set the port (X=1, Y=2) based on settings.
    int port = settings.element_fit.polarisation_type;

    // Construct the output file name based on the settings.
    string output_file = oskar_construct_element_pathname(output_dir, port,
            frequency_hz);

    // Create an element model.
    oskar_Element* element = oskar_element_create(OSKAR_DOUBLE,
            OSKAR_LOCATION_CPU, status);

    // Load the CST text file.
    if (port == 0) port = 1; // FIXME Handle unpolarised input?
    oskar_log_message(log, 0, "Loading CST element pattern: %s", input_file);
    oskar_element_load_cst(element, log, port, input_file,
            &settings.element_fit, status);

    // Save fitted data.
    oskar_element_write(element, port, output_file.c_str(), status);

    // Free memory.
    oskar_element_free(element, status);
    oskar_settings_free(&settings);
}
