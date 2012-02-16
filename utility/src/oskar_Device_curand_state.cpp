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


#include "utility/oskar_Device_curand_state.h"
#include "utility/oskar_device_curand_state_init.h"
#include <cstdlib>
#include <cuda.h>
#include <cstdio>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Device_curand_state::oskar_Device_curand_state(int num_states)
{
    int err = cudaMalloc((void**)&(this->state), num_states * sizeof(curandState));
    this->num_states = num_states;
    if (err != CUDA_SUCCESS)
        throw "Error allocating memory oskar_Device_state::curand_state.";
}

oskar_Device_curand_state::~oskar_Device_curand_state()
{
    if (state != NULL) cudaFree(state);
}

int oskar_Device_curand_state::init(int seed, int offset, int use_device_offset)
{
    return oskar_device_curand_state_init(this->state, this->num_states, 
        seed, offset, use_device_offset);
}


#ifdef __cplusplus
}
#endif
