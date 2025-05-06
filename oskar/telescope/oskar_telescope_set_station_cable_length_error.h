/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_SET_STATION_CABLE_LENGTH_ERROR_H_
#define OSKAR_TELESCOPE_SET_STATION_CABLE_LENGTH_ERROR_H_

/**
 * @file oskar_telescope_set_station_cable_length_error.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets the cable length error for an station in the telescope model.
 *
 * @details
 * This function sets the cable length error of the specified station in
 * the telescope model, transferring data to the GPU if necessary.
 *
 * @param[in,out] telescope    Telescope model to update.
 * @param[in] feed             Feed index (0 = X, 1 = Y).
 * @param[in] index            Station array index to set.
 * @param[in] error_metres     Cable length error, in metres.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_station_cable_length_error(
        oskar_Telescope* telescope,
        int feed,
        int index,
        const double error_metres,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
