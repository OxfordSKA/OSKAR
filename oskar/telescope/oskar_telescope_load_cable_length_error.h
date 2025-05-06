/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_LOAD_CABLE_LENGTH_ERROR_H_
#define OSKAR_TELESCOPE_LOAD_CABLE_LENGTH_ERROR_H_

/**
 * @file oskar_telescope_load_cable_length_error.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads station cable length error data from a text file.
 *
 * @details
 * This function loads element cable length offset data from a comma- or
 * space-separated text file. Each line contains data for one station.
 *
 * @param[in,out] telescope  Telescope model structure to be populated.
 * @param[in] feed           Feed index (0 = X, 1 = Y).
 * @param[in] filename       Name of the data file to load.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_load_cable_length_error(
        oskar_Telescope* telescope,
        int feed,
        const char* filename,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
