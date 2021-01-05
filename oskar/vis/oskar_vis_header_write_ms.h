/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_VIS_HEADER_WRITE_MS_H_
#define OSKAR_VIS_HEADER_WRITE_MS_H_

/**
 * @file oskar_vis_header_write_ms.h
 */

#include <oskar_global.h>
#include <vis/oskar_vis_header.h>
#include <ms/oskar_measurement_set.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Writes visibility header data to a CASA Measurement Set.
 *
 * @details
 * This function writes visibility header data to a CASA Measurement Set
 * and returns a handle to it.
 *
 * @param[in] hdr             Pointer to visibility header structure to write.
 * @param[in] ms_path         Pathname of the Measurement Set to write.
 * @param[in] force_polarised If true, write Stokes I visibility data in
 *                            polarised format, by dividing the power
 *                            equally between XX and YY correlations.
 * @param[in,out] status      Status return code.
 */
OSKAR_APPS_EXPORT
oskar_MeasurementSet* oskar_vis_header_write_ms(const oskar_VisHeader* hdr,
        const char* ms_path, int force_polarised, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
