/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <private_cuda_info.h>
#include <oskar_cuda_info_log.h>
#include <oskar_log.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_cuda_info_log(oskar_Log* log, const oskar_CudaInfo* info)
{
    int i;
    char p = 'M'; /* Log entry priority = message */

    oskar_log_section(log, p, "CUDA System Information");

    /* Print driver and runtime version. */
    oskar_log_value(log, p, 0, "CUDA driver version", "%d.%d",
            info->driver_version / 1000, (info->driver_version % 100) / 10);
    oskar_log_value(log, p, 0, "CUDA runtime version", "%d.%d",
            info->runtime_version / 1000, (info->runtime_version % 100) / 10);
    oskar_log_value(log, p, 0, "Number of CUDA devices detected", "%d",
            info->num_devices);

    /* Print device array. */
    for (i = 0; i < info->num_devices; ++i)
    {
        oskar_log_message(log, p, 0, "Device %d (%s):", i, info->device[i].name);
        oskar_log_value(log, p, 1, "Compute capability", "%d.%d",
                info->device[i].compute.capability.major,
                info->device[i].compute.capability.minor);
        oskar_log_value(log, p, 1, "Supports double precision", "%s",
                info->device[i].supports_double ? "true" : "false");
        oskar_log_value(log, p, 1, "Global memory (MiB)", "%.1f",
                info->device[i].global_memory_size / 1024.0);
        oskar_log_value(log, p, 1, "Free global memory (MiB)", "%.1f",
                info->device[i].free_memory / 1024.0);
        oskar_log_value(log, p, 1, "Number of multiprocessors", "%d",
                info->device[i].num_multiprocessors);
        oskar_log_value(log, p, 1, "Number of CUDA cores", "%d",
                info->device[i].num_cores);
        oskar_log_value(log, p, 1, "GPU clock speed (MHz)", "%.0f",
                info->device[i].gpu_clock / 1000.0);
        oskar_log_value(log, p, 1, "Memory clock speed (MHz)", "%.0f",
                info->device[i].memory_clock / 1000.0);
        oskar_log_value(log, p, 1, "Memory bus width", "%d-bit",
                info->device[i].memory_bus_width);
        oskar_log_value(log, p, 1, "Level-2 cache size (kiB)", "%d",
                info->device[i].level_2_cache_size / 1024);
        oskar_log_value(log, p, 1, "Shared memory size (kiB)", "%d",
                info->device[i].shared_memory_size / 1024);
        oskar_log_value(log, p, 1, "Registers per block", "%d",
                info->device[i].num_registers);
        oskar_log_value(log, p, 1, "Warp size", "%d",
                info->device[i].warp_size);
        oskar_log_value(log, p, 1, "Max threads per block", "%d",
                info->device[i].max_threads_per_block);
        oskar_log_value(log, p, 1, "Max dimensions of a thread block",
                "(%d x %d x %d)",
                info->device[i].max_threads_dim[0],
                info->device[i].max_threads_dim[1],
                info->device[i].max_threads_dim[2]);
        oskar_log_value(log, p, 1, "Max dimensions of a grid",
                "(%d x %d x %d)",
                info->device[i].max_grid_size[0],
                info->device[i].max_grid_size[1],
                info->device[i].max_grid_size[2]);
    }
}

#ifdef __cplusplus
}
#endif
