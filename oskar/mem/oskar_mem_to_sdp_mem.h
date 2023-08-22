/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_TO_SDP_MEM_H_
#define OSKAR_MEM_TO_SDP_MEM_H_

/**
 * @file oskar_mem_to_sdp_mem.h
 */

#include <oskar_global.h>

#ifdef OSKAR_HAVE_SKA_SDP_FUNC
#include "ska-sdp-func/utility/sdp_mem.h"
#else
struct sdp_Mem;
typedef struct sdp_Mem sdp_Mem;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convenience function to return a sdp_Mem from a oskar_Mem pointer.
 *
 * @details
 * This function re-wraps the meta-data in an oskar_Mem to return a handle
 * to a sdp_Mem for use with the SKA SDP processing function library.
 *
 * @param[in] src         Pointer to source memory block.
 * @param[in,out]  status Status return code.
 *
 * @return sdp_Mem* Handle to sdp_Mem.
 */
OSKAR_EXPORT
sdp_Mem* oskar_mem_to_sdp_mem(const oskar_Mem* src, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
