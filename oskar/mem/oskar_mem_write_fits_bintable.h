/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_WRITE_FITS_BINTABLE_H_
#define OSKAR_MEM_WRITE_FITS_BINTABLE_H_

/**
 * @file oskar_mem_write_fits_bintable.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Writes a set of columns to a FITS binary table.
 *
 * @details
 * Writes a set of columns to a FITS binary table.
 * Data in each column is supplied as a pointer to oskar_Mem.
 * The data columns and their names are passed in the variable argument list
 * in pairs, after the \p status parameter. The first item of the pair is the
 * pointer to the column data, and the second item is the name of the
 * column as a string. For example:
 *
 * oskar_Mem* column1 = oskar_mem_create(...);
 * oskar_Mem* column2 = oskar_mem_create(...);
 * size_t num_elements = oskar_mem_length(column1);
 *
 * oskar_mem_write_fits_bintable(
 *         "data.fits", "TABLE_NAME", 2, num_elements, status,
 *         column1, "NAME_OF_COLUMN1", column2, "NAME_OF_COLUMN2"
 * );
 *
 * @param[in] file_name           Name of FITS file to write.
 * @param[in] ext_name            Extension name to write.
 * @param[in] num_mem             Number of columns to write.
 * @param[in] num_elements        Length of each column.
 * @param[in,out] status          Status return code.
 */
OSKAR_EXPORT
void oskar_mem_write_fits_bintable(
        const char* file_name,
        const char* ext_name,
        unsigned int num_mem,
        size_t num_elements,
        int* status,
        ...
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
