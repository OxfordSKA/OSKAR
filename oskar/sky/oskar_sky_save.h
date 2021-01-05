/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_SAVE_H_
#define OSKAR_SKY_SAVE_H_

/**
 * @file oskar_sky_save.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Saves an OSKAR sky model to a text file.
 *
 * @details
 * Saves the specified OSKAR sky model to an ASCII text file.
 * The file contains a simple header, describing the number of sources written,
 * and the data file columns.
 *
 * Note:
 * - The sky model must reside in host (CPU) memory.
 *
 * @param[in] sky         Sky model to write.
 * @param[in] filename    Output filename.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_sky_save(const oskar_Sky* sky, const char* filename, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
