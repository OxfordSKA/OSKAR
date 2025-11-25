/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_COPY_CONTENTS_H_
#define OSKAR_MEM_COPY_CONTENTS_H_

/**
 * @file oskar_mem_copy_contents.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Copies contents of a block of memory into another block of memory.
 *
 * @details
 * This function copies data held in one structure to another structure at a
 * specified element offset.
 *
 * Both data structures must be of the same data type, and there must be enough
 * memory in the destination structure to hold the result: otherwise, an error
 * is returned.
 *
 * @param[out] dst          Pointer to destination data structure to copy into.
 * @param[in]  src          Pointer to source data structure to copy from.
 * @param[in]  offset_dst   Offset into destination memory block.
 * @param[in]  offset_src   Offset from start of source memory block.
 * @param[in]  num_elements Number of elements to copy from source memory block.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_mem_copy_contents(
        oskar_Mem* dst,
        const oskar_Mem* src,
        size_t offset_dst,
        size_t offset_src,
        size_t num_elements,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
