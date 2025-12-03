/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_SET_ALIAS_H_
#define OSKAR_MEM_SET_ALIAS_H_

/**
 * @file oskar_mem_set_alias.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets data for an aliased pointer.
 *
 * @details
 * This function sets the meta-data in a structure to set up a pointer alias
 * to existing memory. The destination structure must not own the memory to
 * which it points, so it must have been created using an
 * oskar_mem_create_alias*() function.
 *
 * @param[in] mem           Handle to destination memory block.
 * @param[in] src           Handle to source memory block.
 * @param[in] offset        Offset number of elements from start of source memory block.
 * @param[in] num_elements  Number of elements in the returned array.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_mem_set_alias(
        oskar_Mem* mem,
        const oskar_Mem* src,
        size_t offset,
        size_t num_elements,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
