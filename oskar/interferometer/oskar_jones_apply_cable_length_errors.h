/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_JONES_APPLY_CABLE_LENGTH_ERRORS_H_
#define OSKAR_JONES_APPLY_CABLE_LENGTH_ERRORS_H_

/**
 * @file oskar_jones_apply_cable_length_errors.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Applies phase errors to a set of Jones matrices.
 *
 * @details
 * Applies phase errors to a set of Jones matrices.
 *
 * Either errors_x or errors_y may be NULL, in which case the complex gain
 * will be treated as 1. If both are NULL, no operation is performed.
 *
 * @param[in,out] jones    Jones matrix block.
 * @param[in] frequency_hz Current channel frequency, in Hz.
 * @param[in] errors_x     Cable length errors, in metres, for X polarisation.
 * @param[in] errors_y     Cable length errors, in metres, for Y polarisation.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_jones_apply_cable_length_errors(
        oskar_Jones* jones,
        double frequency_hz,
        const oskar_Mem* errors_x,
        const oskar_Mem* errors_y,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
