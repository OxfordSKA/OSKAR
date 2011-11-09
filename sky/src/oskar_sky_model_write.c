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


#include "sky/oskar_sky_model_write.h"
#include "sky/oskar_sky_model_check_mem.h"
#include "stdlib.h"
#include "stdio.h"

#ifndef RAD2DEG
#define RAD2DEG 57.295779513082
#endif

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sky_model_write(const char* filename, const oskar_SkyModel* sky)
{
    int i;
    FILE* file;

    if (filename == NULL || sky == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (oskar_sky_model_location(sky) == OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    file = fopen(filename, "w");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    fprintf(file, "# num_sources = %i\n", sky->num_sources);
    fprintf(file, "# RA(deg), Dec(deg), I(Jy), Q(Jy), U(Jy), V(Jy),"
            " ref. freq.(Hz), spectral index\n");
    if (oskar_sky_model_type(sky) == OSKAR_DOUBLE)
    {
        for (i = 0; i < sky->num_sources; ++i)
        {
            fprintf(file, "% -12.6e,% -12.6e,% -12.6e,% -12.6e,% -12.6e,"
                    "% -12.6e,% -12.6e,% -12.6e\n",
                    ((double*)sky->RA.data)[i] * RAD2DEG,
                    ((double*)sky->Dec.data)[i] * RAD2DEG,
                    ((double*)sky->I.data)[i],
                    ((double*)sky->Q.data)[i],
                    ((double*)sky->U.data)[i],
                    ((double*)sky->V.data)[i],
                    ((double*)sky->reference_freq.data)[i],
                    ((double*)sky->spectral_index.data)[i]);
        }
    }
    else if (oskar_sky_model_type(sky) == OSKAR_SINGLE)
    {
        for (i = 0; i < sky->num_sources; ++i)
        {
            fprintf(file, "% -12.6e,% -12.6e,% -12.6e,% -12.6e,% -12.6e,"
                    "% -12.6e,% -12.6e,% -12.6e\n",
                    ((float*)sky->RA.data)[i] * RAD2DEG,
                    ((float*)sky->Dec.data)[i] * RAD2DEG,
                    ((float*)sky->I.data)[i],
                    ((float*)sky->Q.data)[i],
                    ((float*)sky->U.data)[i],
                    ((float*)sky->V.data)[i],
                    ((float*)sky->reference_freq.data)[i],
                    ((float*)sky->spectral_index.data)[i]);
        }
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    fclose(file);
    return 0;
}

#ifdef __cplusplus
}
#endif
