/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_CREATE_H_
#define OSKAR_MEM_CREATE_H_

/**
 * @file oskar_mem_create.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates and initialises a memory block.
 *
 * @details
 * This function creates and initialises an OSKAR memory block, setting the
 * type and location, allocating memory for it as required.
 *
 * A handle to the memory is returned.
 *
 * The memory must be deallocated using oskar_mem_free() when it is
 * no longer required.
 *
 * @param[in] type          Enumerated data type of memory contents.
 * @param[in] location      Either OSKAR_CPU or OSKAR_GPU.
 * @param[in] num_elements  Number of elements of type \p type in the array.
 * @param[in,out]  status   Status return code.
 *
 * @return A handle to the memory block structure.
 */
OSKAR_EXPORT
oskar_Mem* oskar_mem_create(
        int type,
        int location,
        size_t num_elements,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
