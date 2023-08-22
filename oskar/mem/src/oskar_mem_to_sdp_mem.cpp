/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "log/oskar_log.h"
#include "mem/oskar_mem.h"
#include "mem/private_mem.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifdef OSKAR_HAVE_SKA_SDP_FUNC
static sdp_MemType oskar_to_sdp_data_type(const oskar_Mem* src)
{
    int type = SDP_MEM_VOID;
    switch (oskar_mem_precision(src))
    {
    case OSKAR_CHAR:
        type = SDP_MEM_CHAR;
        break;
    case OSKAR_INT:
        type = SDP_MEM_INT;
        break;
    case OSKAR_SINGLE:
        type = SDP_MEM_FLOAT;
        break;
    case OSKAR_DOUBLE:
        type = SDP_MEM_DOUBLE;
        break;
    default:
        break;
    }
    if (oskar_mem_is_complex(src))
    {
        type |= SDP_MEM_COMPLEX;
    }
    return (sdp_MemType) type;
}


static sdp_MemLocation oskar_to_sdp_location(const oskar_Mem* src, int* status)
{
    sdp_MemLocation location = SDP_MEM_CPU;
    switch (oskar_mem_location(src))
    {
    case OSKAR_CPU:
        location = SDP_MEM_CPU;
        break;
    case OSKAR_GPU:
        location = SDP_MEM_GPU;
        break;
    default:
        *status = OSKAR_ERR_BAD_LOCATION;
        oskar_log_error(0, "Memory location not supported by SKA_SDP_FUNC");
        break;
    }
    return location;
}
#endif


sdp_Mem* oskar_mem_to_sdp_mem(const oskar_Mem* src, int* status)
{
#ifdef OSKAR_HAVE_SKA_SDP_FUNC
    sdp_Mem* mem = 0;
    sdp_Error error_status = SDP_SUCCESS;
    int num_dims = 1;
    if (!src)
    {
        mem = sdp_mem_create_wrapper(0,
                SDP_MEM_VOID, SDP_MEM_CPU, 0, 0, 0, &error_status
        );
    }
    else
    {
        const size_t num_elements = oskar_mem_length(src);
        int64_t shape[] = {(int64_t) num_elements, 1, 1};
        const sdp_MemType data_type = oskar_to_sdp_data_type(src);
        const sdp_MemLocation location = oskar_to_sdp_location(src, status);
        if (oskar_mem_is_matrix(src))
        {
            num_dims = 3;
            shape[1] = shape[2] = 2;
        }
        mem = sdp_mem_create_wrapper(
                const_cast<void*>(oskar_mem_void_const(src)),
                data_type, location, num_dims, shape, 0, &error_status
        );
    }
    return mem;
#else
    (void) src;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without support for SKA_SDP_FUNC");
    return 0;
#endif
}

#ifdef __cplusplus
}
#endif
