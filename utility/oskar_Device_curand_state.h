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


#ifndef OSKAR_DEVICE_CURAND_STATE_H_
#define OSKAR_DEVICE_CURAND_STATE_H_

/**
 * @file oskar_Device_curand_state.h
 */

#include "oskar_global.h"
#include <curand.h>
#include <curand_kernel.h>

struct oskar_Device_curand_state
{
    int num_states;     /**< Number of curand states */
    curandState* state; /**< Array of curand states */

#ifdef __cplusplus
    /**
     * @brief Constructor
     *
     * @param[in] num_states Number of curand states to allocate.
     */
    oskar_Device_curand_state(int num_states);

    /**
     * @brief Destructor
     */
    ~oskar_Device_curand_state();

    /**
     * @brief Initialise curand states.
     *
     * @param[in] seed
     * @param[in] offset
     *
     * @return An error code.
     */
    int init(int seed, int offset = 0, int use_device_offset = OSKAR_FALSE);

    /* Convenience pointer casts. */
    operator const curandState*() const { return state; }
    operator curandState*() { return state; }
#endif
};
typedef struct oskar_Device_curand_state oskar_Device_curand_state;


#endif /* OSKAR_DEVICE_CURAND_STATE_H_ */
