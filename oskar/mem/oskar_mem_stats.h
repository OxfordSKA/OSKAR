/*
 * Copyright (c) 2013, The University of Oxford
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

#ifndef OSKAR_MEM_STATS_H_
#define OSKAR_MEM_STATS_H_

/**
 * @file oskar_mem_stats.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Analyses values in a block of memory and reports statistics on them.
 *
 * @details
 * This function analyses values in a block of memory and reports
 * statistics on them.
 *
 * An error is returned if the data type of the memory block is unsupported.
 *
 * @param[in] mem         Pointer to memory block to analyse.
 * @param[in] n           Number of elements to analyse.
 * @param[out] min        The minimum value in the array.
 * @param[out] max        The maximum value in the array.
 * @param[out] mean       The mean value of elements the array.
 * @param[out] std_dev    The population standard deviation of values the array.
 * @param[in,out]  status Status return code.
 */
OSKAR_EXPORT
void oskar_mem_stats(const oskar_Mem* mem, size_t n, double* min, double* max,
        double* mean, double* std_dev, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_STATS_H_ */
