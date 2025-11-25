/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_SAVE_ASCII_H_
#define OSKAR_MEM_SAVE_ASCII_H_

/**
 * @file oskar_mem_save_ascii.h
 */

#include <stdio.h>

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Saves the given blocks of memory to an ASCII table.
 *
 * @details
 * This function saves the given blocks of memory to an ASCII table using the
 * specified stream.
 *
 * The variable argument list must contain pointers to oskar_Mem structures.
 * Data within these structures may reside either in CPU or GPU memory.
 * The number of structures passed is given by the \p num_mem parameter.
 *
 * All structures must contain at least the number of specified
 * \p num_elements. Each array will form one (or more, if using complex types)
 * columns of the output table, with the row corresponding to the element
 * index.
 *
 * @param[in] file          Pointer to output stream.
 * @param[in] num_mem       Number of arrays to write.
 * @param[in] offset        Offset into arrays.
 * @param[in] num_elements  Number of elements to write.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_mem_save_ascii(
        FILE* file,
        size_t num_mem,
        size_t offset,
        size_t num_elements,
        int* status,
        ...
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
