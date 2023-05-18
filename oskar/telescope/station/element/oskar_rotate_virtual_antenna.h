/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_ROTATE_VIRTUAL_ANTENNA_H_
#define OSKAR_ROTATE_VIRTUAL_ANTENNA_H_

/**
 * @file oskar_rotate_virtual_antenna.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to rotate antenna E_theta and E_phi components.
 *
 * @details
 * This function rotates antenna E_theta and E_phi components by the
 * specified angle.
 *
 * @param[in] num_elements   The number of elements in the array.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_rotate_virtual_antenna(int num_elements, int offset,
        double angle_rad, oskar_Mem* beam, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
