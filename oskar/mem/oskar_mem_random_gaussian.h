/*
 * Copyright (c) 2015-2019, The University of Oxford
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

#ifndef OSKAR_MEM_RANDOM_GAUSSIAN_H_
#define OSKAR_MEM_RANDOM_GAUSSIAN_H_

/**
 * @file oskar_mem_random_gaussian.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Generates random numbers from a Gaussian distribution.
 *
 * @details
 * Generates random numbers selected from a Gaussian distribution.
 *
 * This function uses the counter-based stateless "Philox" random number
 * generator from the Random123 library for high performance
 * and crush-resistance.
 *
 * See John K. Salmon et al. "Parallel random numbers: as easy as 1, 2, 3"
 * (doi:10.1145/2063384.2063405)
 *
 * Together, the seed and the counter values completely define the
 * generated data. It is the caller's responsibility to increment the counter
 * value(s) between calls.
 *
 * The array may be either in CPU or GPU memory.
 *
 * @param[in,out] data         Pointer to memory block to fill.
 * @param[in]     seed         Random seed.
 * @param[in]     counter1     User-defined counter.
 * @param[in]     counter2     User-defined counter.
 * @param[in]     counter3     User-defined counter.
 * @param[in]     std          Standard deviation of distribution.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_mem_random_gaussian(oskar_Mem* data, unsigned int seed,
        unsigned int counter1, unsigned int counter2, unsigned int counter3,
        double std, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_RANDOM_GAUSSIAN_H_ */
