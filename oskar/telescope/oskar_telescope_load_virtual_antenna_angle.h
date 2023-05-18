/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_LOAD_VIRTUAL_ANTENNA_ANGLE_H_
#define OSKAR_TELESCOPE_LOAD_VIRTUAL_ANTENNA_ANGLE_H_

/**
 * @file oskar_telescope_load_virtual_antenna_angle.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads the virtual antenna angle file.
 *
 * @details
 * Loads the virtual antenna angle file.
 *
 * @param[in,out] telescope  Telescope model structure to update.
 * @param[in] filename       File name path.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_load_virtual_antenna_angle(oskar_Telescope* telescope,
        const char* filename, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
