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

#include <cuda_runtime_api.h>
#include <cstring>
#include "math/oskar_jones_copy.h"
#include "math/oskar_jones_element_size.h"

extern "C"
int oskar_jones_copy(oskar_Jones* b, const oskar_Jones* a)
{
    // Check that all pointers are not NULL.
    if (a == NULL) return -1;
    if (b == NULL) return -2;
    if (a->data == NULL) return -1;
    if (b->data == NULL) return -2;

    // Get the meta-data.
    int n_sources_a = a->n_sources();
    int n_sources_b = b->n_sources();
    int n_stations_a = a->n_stations();
    int n_stations_b = b->n_stations();
    int type_a = a->type();
    int type_b = b->type();
    int location_a = a->location();
    int location_b = b->location();

    // Check the data dimensions.
    if (n_sources_a != n_sources_b)
        return -11;
    if (n_stations_a != n_stations_b)
        return -12;

    // Check the data types.
    if (type_a != type_b)
        return -100;

    // Get the number of bytes to copy.
    int bytes = oskar_jones_element_size(type_a) * n_sources_a * n_stations_a;

    // Host to host.
    if (location_a == 0 && location_b == 0)
    {
        memcpy(b->data, a->data, bytes);
        return 0;
    }

    // Host to device.
    else if (location_a == 0 && location_b == 1)
    {
        cudaMemcpy(b->data, a->data, bytes, cudaMemcpyHostToDevice);
        return cudaPeekAtLastError();
    }

    // Device to host.
    else if (location_a == 1 && location_b == 0)
    {
        cudaMemcpy(b->data, a->data, bytes, cudaMemcpyDeviceToHost);
        return cudaPeekAtLastError();
    }

    // Device to device.
    else if (location_a == 1 && location_b == 1)
    {
        cudaMemcpy(b->data, a->data, bytes, cudaMemcpyDeviceToDevice);
        return cudaPeekAtLastError();
    }

    return -1000;
}
