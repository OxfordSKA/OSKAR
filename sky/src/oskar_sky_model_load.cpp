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

#include "sky/oskar_sky_model_load.h"

#include <stdio.h>
#include <stdlib.h>
#include "utility/oskar_mem_alloc.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C"
#endif
int oskar_sky_model_load(const char* filename, oskar_SkyModel* sky)
{
    // Check for sane inputs.
    if (filename == NULL || sky == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Open the file.
    FILE* file = fopen(filename, "r");
    if (file == NULL) return OSKAR_ERR_FILE_IO;

    const double deg2rad = 0.0174532925199432957692;
    int type = sky->type();
    char temp_line[1024];
    oskar_SkyModel temp_sky(0, type, OSKAR_LOCATION_CPU);
    int num_sources_loaded = 0;
    if (type == OSKAR_DOUBLE)
    {
        double ra, dec, I, Q, U, V, ref_freq, spectral_index;
        while (fgets(temp_line, sizeof(temp_line), file))
        {
            // Ignore comment lines (lines starting with '#')
            if (temp_line[0] == '#') continue;
            // Load source co-ordinates.
            int read = sscanf(temp_line, "%lf %lf %lf %lf %lf %lf %lf %lf",
                    &ra, &dec, &I, &Q, &U, &V, &ref_freq, &spectral_index);
            if (read != 8) continue;
            // Ensure enough space in arrays.
            if (num_sources_loaded % 100 == 0)
            {
                temp_sky.resize(num_sources_loaded + 100);
            }
            temp_sky.set_source(num_sources_loaded, ra * deg2rad, dec * deg2rad,
                    I, Q, U, V, ref_freq, spectral_index);
            ++(num_sources_loaded);
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        float ra, dec, I, Q, U, V, ref_freq, spectral_index;
        while (fgets(temp_line, sizeof(temp_line), file))
        {
            // Ignore comment lines (lines starting with '#')
            if (temp_line[0] == '#') continue;
            // Load source co-ordinates.
            int read = sscanf(temp_line, "%f %f %f %f %f %f %f %f",
                    &ra, &dec, &I, &Q, &U, &V, &ref_freq, &spectral_index);
            if (read != 8) continue;
            // Ensure enough space in arrays.
            if (num_sources_loaded % 100 == 0)
            {
                temp_sky.resize(num_sources_loaded + 100);
            }
            temp_sky.set_source(num_sources_loaded, ra * deg2rad, dec * deg2rad,
                    I, Q, U, V, ref_freq, spectral_index);
            ++(num_sources_loaded);
        }
    }
    else
    {
        fclose(file);
        return OSKAR_ERR_UNKNOWN;
    }

    temp_sky.num_sources = num_sources_loaded;
    sky->append(&temp_sky);

    fclose(file);
    return 0;
}
















// ====== DEPRECATED =========================================================
#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_model_load_d(const char* file_path, oskar_SkyModelGlobal_d* sky)
{
    // Open the file.
    FILE* file = fopen(file_path, "r");
    if (file == NULL) return;

    const double deg2rad = 0.0174532925199432957692;
    sky->num_sources = 0;
    sky->RA  = NULL;
    sky->Dec = NULL;
    sky->I   = NULL;
    sky->Q   = NULL;
    sky->U   = NULL;
    sky->V   = NULL;
    sky->reference_freq = NULL;
    sky->spectral_index = NULL;

    double ra, dec, I, Q, U, V, ref_freq, spectral_index;

    char line[1024];
    while (fgets(line, sizeof(line), file))
    {
        // Ignore comment lines (lines starting with '#')
        if (line[0] == '#') continue;

        // Load source co-ordinates.
        int read = sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf",
                &ra, &dec, &I, &Q, &U, &V, &ref_freq, &spectral_index);

        if (read != 8) continue;

        // Convert coordinates to radians.
        ra  *= deg2rad;
        dec *= deg2rad;

        // Ensure enough space in arrays.
        if (sky->num_sources % 100 == 0)
        {
            size_t mem_size = ((sky->num_sources) + 100) * sizeof(double);
            sky->RA  = (double*) realloc(sky->RA,  mem_size);
            sky->Dec = (double*) realloc(sky->Dec, mem_size);
            sky->I   = (double*) realloc(sky->I,   mem_size);
            sky->Q   = (double*) realloc(sky->Q,   mem_size);
            sky->U   = (double*) realloc(sky->U,   mem_size);
            sky->V   = (double*) realloc(sky->V,   mem_size);
            sky->reference_freq = (double*) realloc(sky->reference_freq, mem_size);
            sky->spectral_index = (double*) realloc(sky->spectral_index, mem_size);
        }

        sky->RA[sky->num_sources]  = ra;
        sky->Dec[sky->num_sources] = dec;
        sky->I[sky->num_sources]   = I;
        sky->Q[sky->num_sources]   = Q;
        sky->U[sky->num_sources]   = U;
        sky->V[sky->num_sources]   = V;
        sky->reference_freq[sky->num_sources] = ref_freq;
        sky->spectral_index[sky->num_sources] = spectral_index;
        sky->num_sources++;
    }
    fclose(file);
}



void oskar_sky_model_load_f(const char* file_path, oskar_SkyModelGlobal_f* sky)
{
    // Open the file.
    FILE* file = fopen(file_path, "r");
    if (file == NULL) return;

    const float deg2rad = 0.0174532925199432957692f;
    sky->num_sources = 0;
    sky->RA  = NULL;
    sky->Dec = NULL;
    sky->I   = NULL;
    sky->Q   = NULL;
    sky->U   = NULL;
    sky->V   = NULL;
    sky->reference_freq = NULL;
    sky->spectral_index = NULL;

    float ra, dec, I, Q, U, V, spectral_index, ref_freq;

    char line[1024];
    while (fgets(line, sizeof(line), file))
    {
        // Ignore comment lines (lines starting with '#')
        if (line[0] == '#') continue;

        // Load source co-ordinates.
        int read = sscanf(line, "%f %f %f %f %f %f %f %f",
                &ra, &dec, &I, &Q, &U, &V, &ref_freq, &spectral_index);

        if (read != 8) continue;

        // Convert coordinates to radians.
        ra  *= deg2rad;
        dec *= deg2rad;

        // Ensure enough space in arrays.
        if (sky->num_sources % 100 == 0)
        {
            size_t mem_size = ((sky->num_sources) + 100) * sizeof(float);
            sky->RA  = (float*) realloc(sky->RA,  mem_size);
            sky->Dec = (float*) realloc(sky->Dec, mem_size);
            sky->I   = (float*) realloc(sky->I,   mem_size);
            sky->Q   = (float*) realloc(sky->Q,   mem_size);
            sky->U   = (float*) realloc(sky->U,   mem_size);
            sky->V   = (float*) realloc(sky->V,   mem_size);
            sky->reference_freq = (float*) realloc(sky->reference_freq, mem_size);
            sky->spectral_index = (float*) realloc(sky->spectral_index, mem_size);
        }

        sky->RA[sky->num_sources]  = ra;
        sky->Dec[sky->num_sources] = dec;
        sky->I[sky->num_sources]   = I;
        sky->Q[sky->num_sources]   = Q;
        sky->U[sky->num_sources]   = U;
        sky->V[sky->num_sources]   = V;
        sky->reference_freq[sky->num_sources] = ref_freq;
        sky->spectral_index[sky->num_sources] = spectral_index;
        sky->num_sources++;
    }
    fclose(file);
}



#ifdef __cplusplus
}
#endif
