/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_EVALUATE_MAGNETIC_FIELD_H_
#define OSKAR_TELESCOPE_EVALUATE_MAGNETIC_FIELD_H_

/**
 * @file oskar_telescope_evaluate_magnetic_field.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluate Earth magnetic field at all stations in the telescope model.
 *
 * @details
 * Stations in the telescope model are updated with the Earth magnetic field
 * components at the specified date.
 *
 * @param[in,out] telescope Telescope model to update.
 * @param[in] year Epoch at which to evaluate Earth magnetic field model.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_evaluate_magnetic_field(
        oskar_Telescope* telescope,
        double year,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
