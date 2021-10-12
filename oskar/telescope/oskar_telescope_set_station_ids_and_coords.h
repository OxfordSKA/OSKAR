/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_SET_STATION_IDS_AND_COORDS_H_
#define OSKAR_TELESCOPE_SET_STATION_IDS_AND_COORDS_H_

/**
 * @file oskar_telescope_set_station_ids_and_coords.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets station IDs and station coordinates.
 *
 * @details
 * Sets station IDs and station coordinates, recursively if necessary.
 * This prepares the telescope model for use by setting up the
 * station model data.
 *
 * @param[in,out] model         Telescope model structure to update.
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_station_ids_and_coords(
        oskar_Telescope* model, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
