/*
 * Copyright (c) 2011-2014, The University of Oxford
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
#include <cuda_runtime_api.h>
#endif

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device_utils.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_free(oskar_Mem* mem, int* status)
{
    /* Will be safe to call with null pointers. */
    if (!mem) return;

    /* Must proceed with trying to free the memory, regardless of the
     * status code value. */

#ifdef OSKAR_HAVE_OPENCL
    /* Free OpenCL memory if there is a buffer object here. */
    /* This should also be OK for aliases (sub-buffers) as they are
     * reference-counted. */
    if (mem->location & OSKAR_CL)
        clReleaseMemObject(mem->buffer);

    /* For bare pointers, free the memory if the structure actually owns it. */
    else
#endif
    if (mem->owner && mem->data)
    {
        /* Check whether the memory is on the host or the device. */
        if (mem->location == OSKAR_CPU)
        {
            /* Free host memory. */
            free(mem->data);
        }
        else if (mem->location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            /* Free GPU memory. */
            cudaFree(mem->data);
            oskar_device_check_error(status);
#else
            if (status) *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
    }

    /* Free the structure itself. */
    free(mem);
}

#ifdef __cplusplus
}
#endif
