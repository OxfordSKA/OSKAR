/*
 * Copyright (c) 2013-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_LOAD_ASCII_H_
#define OSKAR_MEM_LOAD_ASCII_H_

/**
 * @file oskar_mem_load_ascii.h
 */

#include <stddef.h>

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads an ASCII table from file to populate the given blocks of memory.
 *
 * @details
 * This function reads an ASCII table and populates the supplied arrays
 * from columns in the table.
 *
 * If the default is a blank string, then it is a required column.
 *
 * Note that, for a single column file (with no default), the default must
 * be passed as a blank string, i.e. "".
 *
 * Data within oskar_Mem structures may reside either in CPU or GPU memory.
 * The number of structures passed is given by the \p num_mem parameter.
 *
 * @param[in] filename      Pathname of file to read.
 * @param[in] num_mem       Number of arrays passed to this function.
 * @param[in,out] mem_array Array of memory blocks to fill.
 * @param[in] defaults      Default to use for each array.
 * @param[in,out]  status   Status return code.
 *
 * @return The number of rows read from the file.
 */
OSKAR_EXPORT
size_t oskar_mem_load_ascii_table(
        const char* filename,
        size_t num_mem,
        oskar_Mem** mem_array,
        const char** defaults,
        int* status
);

/**
 * @brief
 * Loads an ASCII table from file to populate the given blocks of memory.
 *
 * @details
 * This function reads an ASCII table and populates the supplied arrays
 * from columns in the table.
 *
 * The variable argument list must contain, for each array, a pointer to an
 * oskar_Mem structure, and a string containing the default value for elements
 * in that array. These parameters alternate throughout the list, so they would
 * appear as: oskar_Mem*, const char*, oskar_Mem*, const char* ...
 *
 * If the default is a blank string, then it is a required column.
 *
 * Note that, for a single column file (with no default), the default must
 * be passed as a blank string, i.e. "".
 *
 * Data within oskar_Mem structures may reside either in CPU or GPU memory.
 * The number of structures passed is given by the \p num_mem parameter.
 *
 * @param[in] filename      Pathname of file to read.
 * @param[in] num_mem       Number of arrays passed to this function.
 * @param[in,out]  status   Status return code.
 *
 * @return The number of rows read from the file.
 */
OSKAR_EXPORT
size_t oskar_mem_load_ascii(
        const char* filename,
        size_t num_mem,
        int* status,
        ...
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
