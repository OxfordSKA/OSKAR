/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_CREATE_ALIAS_FROM_RAW_H_
#define OSKAR_MEM_CREATE_ALIAS_FROM_RAW_H_

/**
 * @file oskar_mem_create_alias_from_raw.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates an aliased pointer from an existing one.
 *
 * @details
 * This function creates a handle to an OSKAR memory block that contains an
 * aliased pointer to existing memory. The structure does not own the memory
 * to which it points.
 *
 * A handle to the memory is returned. The handle must be deallocated
 * using oskar_mem_free() when it is no longer required.
 *
 * @param[in] ptr           Pointer to existing memory.
 * @param[in] type          Enumerated data type of memory contents.
 * @param[in] location      Either OSKAR_CPU or OSKAR_GPU.
 * @param[in] num_elements  Number of elements of type \p type in the array.
 * @param[in,out]  status   Status return code.
 *
 * @return A handle to the aliased memory block structure.
 */
OSKAR_EXPORT
oskar_Mem* oskar_mem_create_alias_from_raw(
        void* ptr,
        int type,
        int location,
        size_t num_elements,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
