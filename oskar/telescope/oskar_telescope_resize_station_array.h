/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_RESIZE_STATION_ARRAY_H_
#define OSKAR_TELESCOPE_RESIZE_STATION_ARRAY_H_

/**
 * @file oskar_telescope_resize_station_array.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Resizes the station array in a telescope model structure.
 *
 * @details
 * Resizes the station array in a telescope model structure.
 *
 * @param[in,out] telescope     Telescope model structure to resize.
 * @param[in]     size          The new number of station models.
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_resize_station_array(oskar_Telescope* telescope,
        int size, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
