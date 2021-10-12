/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_LOAD_STATION_TYPE_MAP_H_
#define OSKAR_TELESCOPE_LOAD_STATION_TYPE_MAP_H_

/**
 * @file oskar_telescope_load_station_type_map.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads a file to specify the station types in a telescope model.
 *
 * @details
 * A telescope station type file is an ASCII text file containing
 * a single column of integer values to indicate the directory index to use
 * for each station in the telescope model.
 *
 * @param[in,out] telescope  Telescope model structure to be updated.
 * @param[in] filename       File name path to a station type file.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_load_station_type_map(oskar_Telescope* telescope,
        const char* filename, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
