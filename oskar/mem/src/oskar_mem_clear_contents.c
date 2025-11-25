/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>
#include <string.h>

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include "log/oskar_log.h"
#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif


void oskar_mem_clear_contents(oskar_Mem* mem, int* status)
{
    if (*status || mem->num_elements == 0) return;
    const size_t size = (
            mem->num_elements * oskar_mem_element_size(mem->type)
    );
    if (mem->location == OSKAR_CPU)
    {
        memset(mem->data, 0, size);
    }
    else if (mem->location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        cudaMemset(mem->data, 0, size);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;           /* LCOV_EXCL_LINE */
#endif
    }
    else if (mem->location & OSKAR_CL)
    {
#ifdef OSKAR_HAVE_OPENCL
        cl_event event = 0;
        cl_int error = 0;
        char zero = '\0';
        error = clEnqueueFillBuffer(
                oskar_device_queue_cl(),
                mem->buffer, &zero, sizeof(char), 0, size, 0, NULL, &event
        );
        clWaitForEvents(1, &event); /* This is required. */
        if (error != CL_SUCCESS)
        {
            oskar_log_error(                              /* LCOV_EXCL_LINE */
                    0,                                    /* LCOV_EXCL_LINE */
                    "clEnqueueFillBuffer() error (%d)",   /* LCOV_EXCL_LINE */
                    error                                 /* LCOV_EXCL_LINE */
            );
            *status = OSKAR_ERR_INVALID_ARGUMENT;         /* LCOV_EXCL_LINE */
        }
#else
        *status = OSKAR_ERR_OPENCL_NOT_AVAILABLE;         /* LCOV_EXCL_LINE */
#endif
    }
    else                                                  /* LCOV_EXCL_LINE */
    {
        *status = OSKAR_ERR_BAD_LOCATION;                 /* LCOV_EXCL_LINE */
    }
}

#ifdef __cplusplus
}
#endif
