/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_STATION_EVALUATE_MAGNETIC_FIELD_H_
#define OSKAR_STATION_EVALUATE_MAGNETIC_FIELD_H_

/**
 * @file oskar_station_evaluate_magnetic_field.h
 */

#include <oskar_global.h>
#include <telescope/station/oskar_station.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluate Earth magnetic field at the station location for the supplied date.
 *
 * @details
 * The station model is updated with the Earth magnetic field components
 * at the specified date.
 *
 * The field is currently evaluated using an IGRF14 routine, which is
 * valid until the year 2035.
 *
 * @param[in,out] station         Station model.
 * @param[in] year                Date at which to evaluate field model.
 * @param[in,out] status          Status return code.
 */
OSKAR_EXPORT
void oskar_station_evaluate_magnetic_field(
        oskar_Station* station,
        double year,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
