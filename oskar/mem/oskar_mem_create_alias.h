/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_CREATE_ALIAS_H_
#define OSKAR_MEM_CREATE_ALIAS_H_

/**
 * @file oskar_mem_create_alias.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Deprecated. Creates an aliased pointer from an existing one.
 *
 * @details
 * @note This function is deprecated.
 *
 * This function creates a handle to a memory block that contains an
 * aliased pointer to (part of) an existing memory block. The structure does
 * not own the memory to which it points.
 *
 * A handle to the memory is returned. The handle must be deallocated
 * using oskar_mem_free() when it is no longer required.
 *
 * @param[in] src           Handle to source memory block (may be NULL).
 * @param[in] offset        Offset number of elements from start of source memory block.
 * @param[in] num_elements  Number of elements in the returned array.
 * @param[in,out]  status   Status return code.
 *
 * @return A handle to the aliased memory.
 */
OSKAR_EXPORT
oskar_Mem* oskar_mem_create_alias(
        const oskar_Mem* src,
        size_t offset,
        size_t num_elements,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
