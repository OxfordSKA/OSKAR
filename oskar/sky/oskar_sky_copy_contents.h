/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_COPY_CONTENTS_H_
#define OSKAR_SKY_COPY_CONTENTS_H_

/**
 * @file oskar_sky_copy_contents.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Copies source information from one sky model into another.
 *
 * @details
 * Note: this function does not alter meta-data (num_sources, and use_extended)
 * fields of the destination model.
 *
 * @param[out] dst         Sky model to copy into.
 * @param[in]  src         Sky model to copy from.
 * @param[in]  offset_dst  Required offset into the destination sky model.
 * @param[in]  offset_src  Offset from start of source sky model.
 * @param[in]  num_sources Number of sources to copy from source sky model.
 * @param[in,out] status   Status return code.
*/
OSKAR_EXPORT
void oskar_sky_copy_contents(
        oskar_Sky* dst,
        const oskar_Sky* src,
        int offset_dst,
        int offset_src,
        int num_sources,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
