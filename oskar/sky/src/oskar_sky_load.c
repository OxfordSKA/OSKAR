/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <oskar_sky.h>
#include <oskar_getline.h>
#include <oskar_string_to_array.h>

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

static const double deg2rad = 1.74532925199432957692369e-2;
static const double arcsec2rad = 4.84813681109535993589914e-6;

oskar_Sky* oskar_sky_load(const char* filename, int type, int* status)
{
    int n = 0;
    FILE* file;
    char* line = 0;
    size_t bufsize = 0;
    oskar_Sky* sky;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Get the data type. */
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }

    /* Initialise the sky model. */
    sky = oskar_sky_create(type, OSKAR_CPU, 0, status);

    /* Loop over lines in file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        /* Set defaults. */
        /* RA, Dec, I, Q, U, V, freq0, spix, RM, FWHM maj, FWHM min, PA */
        double par[] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
        size_t num_param = sizeof(par) / sizeof(double);
        size_t num_required = 3, num_read = 0;

        /* Load source parameters (require at least RA, Dec, Stokes I). */
        num_read = oskar_string_to_array_d(line, num_param, par);
        if (num_read < num_required)
            continue;

        /* Ensure enough space in arrays. */
        if (oskar_sky_num_sources(sky) <= n)
        {
            oskar_sky_resize(sky, n + 100, status);
            if (*status)
                break;
        }

        if (num_read <= 9)
        {
            /* RA, Dec, I, Q, U, V, freq0, spix, RM */
            oskar_sky_set_source(sky, n, par[0] * deg2rad,
                    par[1] * deg2rad, par[2], par[3], par[4], par[5],
                    par[6], par[7], par[8], 0.0, 0.0, 0.0, status);
        }
        else if (num_read == 11)
        {
            /* Old format, with no rotation measure. */
            /* RA, Dec, I, Q, U, V, freq0, spix, FWHM maj, FWHM min, PA */
            oskar_sky_set_source(sky, n, par[0] * deg2rad,
                    par[1] * deg2rad, par[2], par[3], par[4], par[5],
                    par[6], par[7], 0.0, par[8] * arcsec2rad,
                    par[9] * arcsec2rad, par[10] * deg2rad, status);
        }
        else if (num_read == 12)
        {
            /* New format. */
            /* RA, Dec, I, Q, U, V, freq0, spix, RM, FWHM maj, FWHM min, PA */
            oskar_sky_set_source(sky, n, par[0] * deg2rad,
                    par[1] * deg2rad, par[2], par[3], par[4], par[5],
                    par[6], par[7], par[8], par[9] * arcsec2rad,
                    par[10] * arcsec2rad, par[11] * deg2rad, status);
        }
        else
        {
            /* Error. */
            *status = OSKAR_ERR_BAD_SKY_FILE;
            break;
        }
        ++n;
    }

    /* Set the size to be the actual number of elements loaded. */
    oskar_sky_resize(sky, n, status);

    /* Free the line buffer and close the file. */
    if (line) free(line);
    fclose(file);

    /* Check if an error occurred. */
    if (*status)
    {
        oskar_sky_free(sky, status);
        sky = 0;
    }

    /* Return a handle to the sky model. */
    return sky;
}

#ifdef __cplusplus
}
#endif
