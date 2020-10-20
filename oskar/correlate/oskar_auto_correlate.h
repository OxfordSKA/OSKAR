/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_AUTO_CORRELATE_H_
#define OSKAR_AUTO_CORRELATE_H_

/**
 * @file oskar_auto_correlate.h
 */

#include <oskar_global.h>
#include <interferometer/oskar_jones.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Multiply a set of Jones matrices with a set of source brightness
 * matrices to form visibilities (i.e. V = J B J*).
 *
 * @details
 * The source brightness matrices are constructed from the supplied
 * Stokes parameters.
 *
 * @param[in]  num_sources  Number of sources to use.
 * @param[in]  jones        Set of Jones matrices.
 * @param[in]  src_flux[4]  Vectors of source Stokes (I, Q, U, V) values.
 * @param[out] offset_out   Start offset into output array.
 * @param[out] vis          Output visibilities.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_auto_correlate(
        int num_sources,
        const oskar_Jones* jones,
        const oskar_Mem* const src_flux[4],
        int offset_out,
        oskar_Mem* vis,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
