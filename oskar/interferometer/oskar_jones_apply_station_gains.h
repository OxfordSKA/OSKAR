/*
 * Copyright (c) 2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_JONES_APPLY_STATION_GAINS_H_
#define OSKAR_JONES_APPLY_STATION_GAINS_H_

/**
 * @file oskar_jones_apply_station_gains.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Applies station gains to a set of Jones matrices.
 *
 * @details
 * Applies station gains to a set of Jones matrices.
 *
 * @param[in,out] jones  Jones matrix block.
 * @param[in]     gains  Vector of station gains.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
void oskar_jones_apply_station_gains(oskar_Jones* jones,
        oskar_Mem* gains, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
