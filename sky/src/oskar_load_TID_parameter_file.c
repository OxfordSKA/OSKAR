/*
 * Copyright (c) 2013, The University of Oxford
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


#include "sky/oskar_load_TID_parameter_file.h"

#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_load_TID_parameter_file(oskar_SettingsTIDscreen* TID,
        const char* filename, int* status)
{
    /* Declare the line buffer and counter. */
    char* line = NULL;
    size_t bufsize = 0;
    int n = 0, type = 0;
    FILE* file;

    TID->num_components = 0;
    TID->amp = NULL;
    TID->speed = NULL;
    TID->wavelength = NULL;
    TID->theta = NULL;

    /* Check all inputs. */
    if (!TID || !filename || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Loop over each line in the file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        int read = 0;

//        printf("[%i] %s\n", n, line);

        /* Ignore comment lines (lines starting with '#'). */
        if (line[0] == '#') continue;

        /* Read the screen height */
        if (n == 0)
        {
            read = oskar_string_to_array_d(line, 1, &TID->height_km);
            if (read != 1) continue;
//            printf("---> reading height...\n");
            ++n;
        }
//        /* Read the TEC0 value */
//        else if (n == 1)
//        {
//            read = oskar_string_to_array_d(line, 1, &TID->TEC0);
//            if (read != 1) continue;
////            printf("---> reading TEC0...\n");
//            ++n;
//        }
        /* Read TID components */
        else
        {
            double par[] = {0.0, 0.0, 0.0, 0.0};
            read = oskar_string_to_array_d(line, sizeof(par)/sizeof(double), par);
            if (read != 4) continue;
//            printf("---> reading component... %i\n", TID->num_components);

            /* Resize component arrays. */
            size_t newSize = (TID->num_components+1) * sizeof(double);
            TID->amp = (double*)realloc(TID->amp, newSize);
            TID->speed = (double*)realloc(TID->speed, newSize);
            TID->wavelength = (double*)realloc(TID->wavelength, newSize);
            TID->theta = (double*)realloc(TID->theta, newSize);

            /* Store the component */
            TID->amp[TID->num_components] = par[0];
            TID->speed[TID->num_components] = par[1];
            TID->theta[TID->num_components] = par[2];
            TID->wavelength[TID->num_components] = par[3];
            ++n;
            ++(TID->num_components);
        }
    }

    /* Free the line buffer and close the file. */
    if (line) free(line);
    fclose(file);
}


#ifdef __cplusplus
}
#endif
