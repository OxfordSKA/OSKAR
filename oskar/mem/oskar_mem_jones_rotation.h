/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_JONES_ROTATION_H_
#define OSKAR_MEM_JONES_ROTATION_H_

/**
 * @file oskar_mem_jones_rotation.h
 */

#include <stddef.h>
#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Rotate all Jones matrix elements in an array by the given angle.
 *
 * @details
 * Rotate all Jones matrix elements in an array by the given angle.
 *
 * @param[in,out]  mem          Array to rotate.
 * @param[in]      angle_rad    Angle by which to rotate elements, in radians.
 * @param[in]      offset       Start offset into array.
 * @param[in]      num_elements Number of elements to scale.
 * @param[in,out]  status       Status return code.
 */
OSKAR_EXPORT
void oskar_mem_jones_rotation(
        oskar_Mem* mem,
        double angle_rad,
        size_t offset,
        size_t num_elements,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
