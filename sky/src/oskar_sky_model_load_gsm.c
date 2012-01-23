/*
 * Copyright (c) 2011, The University of Oxford
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

#include "math/oskar_healpix_pix_to_angles_ring.h"
#include "sky/oskar_galactic_to_fk5.h"
#include "sky/oskar_sky_model_append.h"
#include "sky/oskar_sky_model_init.h"
#include "sky/oskar_sky_model_load.h"
#include "sky/oskar_sky_model_resize.h"
#include "sky/oskar_sky_model_set_source.h"
#include "sky/oskar_sky_model_type.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sky_model_load_gsm(oskar_SkyModel* sky, const char* filename,
        int nside)
{
    int type, n = 0;
    FILE* file;
    char* line = NULL;
    size_t bufsize = 0;
    oskar_SkyModel temp_sky;

    /* Check for sane inputs. */
    if (sky == NULL || filename == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Open the file. */
    file = fopen(filename, "r");
    if (file == NULL) return OSKAR_ERR_FILE_IO;

    /* Get the data type. */
    type = oskar_sky_model_type(sky);
    oskar_sky_model_init(&temp_sky, type, OSKAR_LOCATION_CPU, 0);

    if (type == OSKAR_DOUBLE)
    {
        while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
        {
            double l, b, ra, dec, par = 0.0;

            /* Ignore comment lines (lines starting with '#') */
            if (line[0] == '#') continue;

            /* Load pixel value. */
            if (oskar_string_to_array_d(line, 1, &par) < 1) continue;

            /* Ensure enough space in arrays. */
            if (n % 100 == 0)
            {
                int err;
                err = oskar_sky_model_resize(&temp_sky, n + 100);
                if (err)
                {
                    fclose(file);
                    return err;
                }
            }
            
            /* Compute Galactic longitude and latitude from pixel index. */
            oskar_healpix_pix_to_angles_ring(nside, n, &b, &l);
            b = (M_PI / 2.0) - b; /* Colatitude to latitude. */
            
            /* Compute RA and Dec. */
            oskar_galactic_to_fk5_d(1, &l, &b, &ra, &dec);
            
            /* Store pixel data. */
            oskar_sky_model_set_source(&temp_sky, n,
                    ra, dec, par, 0.0, 0.0, 0.0, 0.0, 0.0);
            ++n;
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
        {
            double l, b;
            float tl, tb, ra, dec, par = 0.0;

            /* Ignore comment lines (lines starting with '#') */
            if (line[0] == '#') continue;

            /* Load pixel value. */
            if (oskar_string_to_array_f(line, 1, &par) < 1) continue;

            /* Ensure enough space in arrays. */
            if (n % 100 == 0)
            {
                int err;
                err = oskar_sky_model_resize(&temp_sky, n + 100);
                if (err)
                {
                    fclose(file);
                    return err;
                }
            }
            
            /* Compute Galactic longitude and latitude from pixel index. */
            oskar_healpix_pix_to_angles_ring(nside, n, &b, &l);
            b = (M_PI / 2.0) - b; /* Colatitude to latitude. */
            
            /* Compute RA and Dec. */
            tl = l; tb = b;
            oskar_galactic_to_fk5_f(1, &tl, &tb, &ra, &dec);
            
            /* Store pixel data. */
            oskar_sky_model_set_source(&temp_sky, n, (float)ra,
                    (float)dec, (float)par, 0.0, 0.0, 0.0, 0.0, 0.0);
            ++n;
        }
    }
    else
    {
        fclose(file);
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    /* Record the number of elements loaded. */
    temp_sky.num_sources = n;
    oskar_sky_model_append(sky, &temp_sky);

    /* Free the line buffer and close the file. */
    if (line) free(line);
    fclose(file);

    return 0;
}

#ifdef __cplusplus
}
#endif
