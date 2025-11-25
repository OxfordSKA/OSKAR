/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_LOAD_NAMED_COLUMNS_H_
#define OSKAR_SKY_LOAD_NAMED_COLUMNS_H_

/**
 * @file oskar_sky_load_named_columns.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads source data from a formatted text file with named columns.
 *
 * @details
 * This function loads a text file conforming to the LOFAR/BBS/DP3/WSClean
 * sky model format, described at
 * https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb#format_string
 *
 * Column types are defined using a "Format = " line, which should
 * appear at the top of the file before the data table.
 *
 * Data in the columns may be space and/or comma separated.
 *
 * Text appearing on a line after a hash symbol (#) is treated as a comment,
 * and is therefore ignored. The only exception is the "Format = " statement,
 * which may appear inside a header comment.
 *
 * @param[in]  filename  Path to a source list text file.
 * @param[in]  type      Required data type (OSKAR_SINGLE or OSKAR_DOUBLE).
 * @param[in,out] status Status return code.
 *
 * @return A handle to the sky model, or NULL if an error occurred.
 */
OSKAR_EXPORT
oskar_Sky* oskar_sky_load_named_columns(
        const char* filename,
        int type,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
