/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#ifdef OSKAR_HAVE_CUDA
/* Must include this first to avoid type conflict.*/
#include <cuda_runtime_api.h>
#define H2D cudaMemcpyHostToDevice
#endif

#include <stdlib.h>
#include <math.h>

#include <private_station.h>
#include <oskar_station.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_set_element_type(oskar_Station* dst,
        int index, int element_type, int* status)
{
    int location, *t_;
    size_t offset_bytes;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check range. */
    if (index >= dst->num_elements)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Get the data type. */
    location = oskar_mem_location(dst->element_types);
    offset_bytes = index * sizeof(int);

    /* Set the data. */
    oskar_mem_int(dst->element_types_cpu, status)[index] = element_type;
    t_ = oskar_mem_int(dst->element_types, status);
    if (location == OSKAR_CPU)
        t_[index] = element_type;
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        cudaMemcpy((char*)t_ + offset_bytes, &element_type, sizeof(int), H2D);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
