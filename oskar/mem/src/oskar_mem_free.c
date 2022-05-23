/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_free(oskar_Mem* mem, int* status)
{
    /* Will be safe to call with null pointers. */
    if (!mem) return;

    /* Decrement reference count and return if there are still references. */
    oskar_mutex_lock(mem->mutex);
    mem->ref_count--;
    oskar_mutex_unlock(mem->mutex);
    if (mem->ref_count > 0)
    {
        return;
    }

    /* Must proceed with trying to free the memory, regardless of the
     * status code value. */

#ifdef OSKAR_HAVE_OPENCL
    /* Free OpenCL memory if there is a buffer object here. */
    /* This should also be OK for aliases (sub-buffers) as they are
     * reference-counted. */
    if (mem->location & OSKAR_CL)
    {
        if (mem->buffer)
        {
            clReleaseMemObject(mem->buffer);
        }
    }
#endif
    /* For bare pointers, free the memory if the structure actually owns it. */
    if (mem->owner && mem->data && !(mem->location & OSKAR_CL))
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
            const int error = (int)cudaFree(mem->data);
            if (status && error) *status = error;
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
