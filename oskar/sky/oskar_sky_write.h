/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_WRITE_H_
#define OSKAR_SKY_WRITE_H_

/**
 * @file oskar_sky_write.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Writes an OSKAR sky model to a binary file.
 *
 * @details
 * Writes the specified OSKAR sky model to a binary file.
 *
 * @param[in] sky         Sky model to write.
 * @param[in] filename    Output filename.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_sky_write(const oskar_Sky* sky, const char* filename, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
