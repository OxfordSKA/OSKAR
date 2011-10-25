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

#include "oskar_global.h"
#include "interferometry/oskar_Visibilities.h"
#include "utility/oskar_mem_element_size.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

#ifdef __cplusplus
extern "C"
#endif
int oskar_visibilties_read(oskar_Visibilities* vis, const char* filename)
{
    if (filename == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // TODO some checks on the validity of the visibility data.
    // TODO require vis to be loaded to CPU?

    // Open the file to write to.
    FILE* file;
    file = fopen(filename, "rb");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    int num_times = 0;
    int num_baselines = 0;
    int num_channels = 0;
    int coord_type = 0;
    int amp_type = 0;

    // Read header.
    // TODO check return values of fread().
    if (fread(&num_times, sizeof(int), 1, file) != 1)
    {
        fclose(file);
        return OSKAR_ERR_FILE_IO;
    }
    if (fread(&num_baselines, sizeof(int), 1, file) != 1)
    {
        fclose(file);
        return OSKAR_ERR_FILE_IO;
    }
    if (fread(&num_channels, sizeof(int), 1, file) != 1)
    {
        fclose(file);
        return OSKAR_ERR_FILE_IO;
    }
    if (fread(&coord_type, sizeof(int), 1, file) != 1)
    {
        fclose(file);
        return OSKAR_ERR_FILE_IO;
    }
    if (fread(&amp_type, sizeof(int), 1, file) != 1)
    {
        fclose(file);
        return OSKAR_ERR_FILE_IO;
    }

    // Resize the visibility data to match the header (reallocate if needed)
    // TODO this needs a resize method?
    if (vis != NULL) delete vis;
    vis = new oskar_Visibilities(num_times, num_baselines, num_channels,
            amp_type, OSKAR_LOCATION_CPU);

    // Read data.
    // TODO check return values of fread().
    size_t num_samples = vis->num_samples();
    size_t coord_element_size = oskar_mem_element_size(coord_type);
    size_t amp_element_size = oskar_mem_element_size(amp_type);
    if (fread(vis->baseline_u.data, coord_element_size, num_samples, file) != num_samples)
    {
        fclose(file);
        return OSKAR_ERR_FILE_IO;
    }
    if (fread(vis->baseline_v.data, coord_element_size, num_samples, file) != num_samples)
    {
        fclose(file);
        return OSKAR_ERR_FILE_IO;
    }

    if (fread(vis->baseline_w.data, coord_element_size, num_samples, file) != num_samples)
    {
        fclose(file);
        return OSKAR_ERR_FILE_IO;
    }

    if (fread(vis->amplitude.data,  amp_element_size,   num_samples, file) != num_samples)
    {
        fclose(file);
        return OSKAR_ERR_FILE_IO;
    }

    fclose(file);

    return 0;
}

