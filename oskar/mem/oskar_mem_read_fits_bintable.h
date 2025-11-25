/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_READ_FITS_BINTABLE_H_
#define OSKAR_MEM_READ_FITS_BINTABLE_H_

/**
 * @file oskar_mem_read_fits_bintable.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Reads a data column from a FITS binary table.
 *
 * @details
 * Reads a data column from a FITS binary table.
 *
 * @param[in] file_name          Name of FITS file to read.
 * @param[in] ext_name           Extension name to search for.
 * @param[in] column_name        Column name to read.
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
oskar_Mem* oskar_mem_read_fits_bintable(
        const char* file_name,
        const char* ext_name,
        const char* column_name,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
