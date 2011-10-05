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

#include "math/oskar_jones_alloc.h"
#include "math/oskar_jones_element_size.h"
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C"
#endif
int oskar_jones_alloc(oskar_Jones* jones)
{
    // Check that the structure exists.
    if (jones == NULL) return -1;

    // Get the meta-data.
#ifdef __cplusplus
    int n_sources = jones->n_sources();
    int n_stations = jones->n_stations();
    int location = jones->location();
    int type = jones->type();
#else
    int n_sources = jones->private_n_sources;
    int n_stations = jones->private_n_stations;
    int location = jones->private_location;
    int type = jones->private_type;
#endif

    // Get the memory size.
    size_t element_size = oskar_jones_element_size(type);
    size_t bytes = n_sources * n_stations * element_size;

    // Check whether the memory should be on the host or the device.
    int err = 0;
    if (location == 0)
    {
        // Allocate host memory.
        jones->data = malloc(bytes);
        if (jones->data == NULL) err = -2;
    }
    else if (location == 1)
    {
        // Allocate GPU memory.
        cudaMalloc(&jones->data, bytes);
        err = cudaPeekAtLastError();
    }
    return err;
}
