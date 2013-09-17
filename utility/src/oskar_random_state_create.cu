/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <private_random_state.h>
#include <oskar_random_state.h>
#include <oskar_cuda_check_error.h>
#include <curand_kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel prototype. */
__global__
static void oskar_random_state_init_cudak(curandStateXORWOW* state,
        int num_states, unsigned long long seed, unsigned long long offset,
        unsigned long long device_offset);

/* Kernel wrappers. ======================================================== */

oskar_RandomState* oskar_random_state_create(int num_states,
        int seed, int offset, int use_device_offset, int* status)
{
    int num_blocks, num_threads = 128, device_offset = 0;
    oskar_RandomState* state = 0;

    /* Check all inputs. */
    if (!status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Allocate the structure. */
    state = (oskar_RandomState*) malloc(sizeof(oskar_RandomState));

    /* Initialise. */
    state->state = 0;
    state->num_states = 0;
    if (num_states == 0)
        return state;

    /* Check if safe to proceed. */
    if (*status) return state;

    /* Allocate memory for states. */
    cudaMalloc((void**)&(state->state), num_states * sizeof(curandStateXORWOW));
    state->num_states = num_states;
    oskar_cuda_check_error(status);

    /* Note: device_offset allocates different states from same seed to span
     * multiple GPUs. */
    if (use_device_offset)
    {
        int device_id = 0;
        cudaGetDevice(&device_id);
        device_offset = device_id * num_states;
    }

    /* Call kernel to initialise states. */
    num_blocks = (num_states + num_threads - 1) / num_threads;
    oskar_random_state_init_cudak OSKAR_CUDAK_CONF(num_blocks, num_threads)
            (state->state, num_states, seed, offset, device_offset);
    oskar_cuda_check_error(status);

    return state;
}


/* Kernels. ================================================================ */

__global__
static void oskar_random_state_init_cudak(curandStateXORWOW* state,
        int num_states, unsigned long long seed, unsigned long long offset,
        unsigned long long device_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;

    /* curand_init(seed==sequence, sub-sequence, offset, state) */
    curand_init(seed, idx + device_offset, offset, &state[idx]);
}

#ifdef __cplusplus
}
#endif
