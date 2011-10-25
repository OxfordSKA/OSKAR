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
int oskar_visibilties_write(const char* filename, const oskar_Visibilities* vis)
{
    if (filename == NULL || vis == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (vis->location() != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    // TODO some checks on the validity of the visibility data.

    // Open the file to write to.
    FILE* file;
    file = fopen(filename, "wb");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    // Local variables.
    int num_samples = vis->num_samples();
    int coord_type = vis->baseline_u.type();
    int amp_type = vis->amplitude.type();
    size_t coord_element_size = oskar_mem_element_size(coord_type);
    size_t amp_element_size = oskar_mem_element_size(amp_type);


    // Write header.
    // TODO check return values from fwrite() ...
    fwrite(&(vis->num_times), sizeof(int), 1, file);
    fwrite(&(vis->num_baselines), sizeof(int), 1, file);
    fwrite(&(vis->num_channels), sizeof(int), 1, file);
    fwrite(&coord_type, sizeof(int), 1, file);
    fwrite(&amp_type, sizeof(int), 1, file);

    // Write data.
    // TODO check return values from fwrite() ...
    fwrite(vis->baseline_u.data, coord_element_size, num_samples, file);
    fwrite(vis->baseline_v.data, coord_element_size, num_samples, file);
    fwrite(vis->baseline_w.data, coord_element_size, num_samples, file);
    fwrite(vis->amplitude.data,  amp_element_size,   num_samples, file);

    fclose(file);
    return 0;
}

