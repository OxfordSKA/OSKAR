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

#include <cuda_runtime_api.h>
#include <oskar_cuda_mem_log.h>
#include <oskar_log.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_cuda_mem_log(oskar_Log* log, int depth, int device_id)
{
    struct cudaDeviceProp device_prop;
    size_t mem_free = 0, mem_total = 0;

    /* Record GPU memory usage. */
    cudaMemGetInfo(&mem_free, &mem_total);
    cudaGetDeviceProperties(&device_prop, device_id);
    oskar_log_message(log, 'M', depth, "Memory on GPU %d [%s] is %.1f%% "
            "(%.1f/%.1f GB) used.",
            device_id, device_prop.name,
            100.0 * (1.0 - (mem_free / (double)mem_total)),
            (mem_total-mem_free)/(1024.*1024.*1024.),
            mem_total/(1024.*1024.*1024.));
}

#ifdef __cplusplus
}
#endif
