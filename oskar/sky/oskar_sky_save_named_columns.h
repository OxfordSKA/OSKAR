/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_SAVE_NAMED_COLUMNS_H_
#define OSKAR_SKY_SAVE_NAMED_COLUMNS_H_

/**
 * @file oskar_sky_save_named_columns.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Saves a sky model to a formatted text file with named columns.
 *
 * @details
 * This function saves a text file conforming to the LOFAR/BBS/DP3/WSClean
 * sky model format, described at
 * https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb#format_string
 *
 * Column types are defined using a "# (...) = format" line, which is written
 * as part of the header at the top of the file before the data table.
 *
 * @param[in] sky         Sky model to write.
 * @param[in] filename    Output filename.
 * @param[in] use_degree_coord_column If true, don't use a suffix for degrees.
 * @param[in] write_name  If true, write a source name, based on its index.
 * @param[in] write_type  If true, write source type (POINT/GAUSSIAN).
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_sky_save_named_columns(
        const oskar_Sky* sky,
        const char* filename,
        int use_degree_coord_column,
        int write_name,
        int write_type,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
