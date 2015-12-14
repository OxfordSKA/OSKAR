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

#include <oskar_settings_load.h>
#include <oskar_settings_load_beam_pattern.h>
#include <oskar_settings_load_image.h>
#include <oskar_settings_load_interferometer.h>
#include <oskar_settings_load_ionosphere.h>
#include <oskar_settings_load_observation.h>
#include <oskar_settings_load_simulator.h>
#include <oskar_settings_load_sky.h>
#include <oskar_settings_load_telescope.h>
#include <oskar_settings_load_element_fit.h>

#include <oskar_settings_file_exists.h>
#include <oskar_settings_init.h>

#include <oskar_log.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_settings_load(oskar_Settings* s, oskar_Log* log,
        const char* filename, int* status)
{
    /* Check if the settings file exists! */
    if (!oskar_settings_file_exists(filename))
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Initialise the settings arrays. */
    oskar_settings_init(s);

    /* Load observation settings first. */
    oskar_settings_load_observation(&s->obs, log, filename, status);

    oskar_settings_load_simulator(&s->sim, filename, status);
    oskar_settings_load_sky(&s->sky, filename, status);
    oskar_settings_load_telescope(&s->telescope, filename, status);
    oskar_settings_load_beam_pattern(&s->beam_pattern, filename, status);
    oskar_settings_load_interferometer(&s->interferometer, filename, status);
    oskar_settings_load_image(&s->image, filename, status);
    oskar_settings_load_element_fit(&s->element_fit, filename, status);
    /* oskar_settings_load_ionosphere(&s->ionosphere, filename, status); */

    /* Save the path to the settings file. */
    s->settings_path = realloc(s->settings_path, 1 + strlen(filename));
    strcpy(s->settings_path, filename);
}

#ifdef __cplusplus
}
#endif
