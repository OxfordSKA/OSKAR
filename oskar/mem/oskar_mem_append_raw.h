/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_APPEND_RAW_H_
#define OSKAR_MEM_APPEND_RAW_H_

/**
 * @file oskar_mem_append_raw.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Appends to a memory block by copying data from another memory location.
 *
 * @details
 * This function copies memory from a raw pointer and appends
 * it to memory held in an oskar_Mem structure.
 *
 * @param[out] to           Pointer to block which will be extended.
 * @param[in] from          Start address of memory to copy.
 * @param[in] from_type     Enumerated type of memory to copy.
 * @param[in] from_location Enumerated location of memory to copy.
 * @param[in] num_elements  Number of elements to copy.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_mem_append_raw(
        oskar_Mem* to,
        const void* from,
        int from_type,
        int from_location,
        size_t num_elements,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
