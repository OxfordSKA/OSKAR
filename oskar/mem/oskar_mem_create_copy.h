/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_CREATE_COPY_H_
#define OSKAR_MEM_CREATE_COPY_H_

/**
 * @file oskar_mem_create_copy.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates and initialises a new memory block by copying another.
 *
 * @details
 * This function creates and initialises a new memory block by copying an
 * existing one to the specified location.
 *
 * A handle to the new memory block is returned.
 *
 * The memory block must be deallocated using oskar_mem_free() when it is
 * no longer required.
 *
 * @param[in] src           Handle to existing memory block to copy.
 * @param[in] location      Enumerated location of new memory block.
 * @param[in,out] status    Status return code.
 *
 * @return A handle to the new memory block.
 */
OSKAR_EXPORT
oskar_Mem* oskar_mem_create_copy(
        const oskar_Mem* src,
        int location,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
