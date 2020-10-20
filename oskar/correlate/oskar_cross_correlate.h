/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CROSS_CORRELATE_H_
#define OSKAR_CROSS_CORRELATE_H_

/**
 * @file oskar_cross_correlate.h
 */

#include <oskar_global.h>
#include <telescope/oskar_telescope.h>
#include <interferometer/oskar_jones.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Multiply a set of Jones matrices with a set of source brightness
 * matrices to form visibilities (i.e. V = J B J*).
 *
 * @details
 * The source brightness matrices are constructed from the supplied
 * Stokes parameters.
 *
 * The Jones matrices should have dimensions corresponding to the number of
 * sources in the brightness matrix and the number of stations.
 *
 * @param[in]  source_type    Source type (0 = point, 1 = Gaussian).
 * @param[in]  num_sources    Number of sources to use.
 * @param[in]  jones          Set of Jones matrices.
 * @param[in]  src_flux[4]    Vectors of source Stokes (I, Q, U, V) values.
 * @param[in]  src_dir[3]     Vectors of source direction cosines.
 * @param[in]  src_ext[3]     Vectors of extended source parameters.
 * @param[in]  tel            Telescope model.
 * @param[in]  station_uvw[3] Station (u, v, w) coordinates, in metres.
 * @param[in]  gast           Greenwich apparent sidereal time, in radians.
 * @param[in]  frequency_hz   Current observation frequency, in Hz.
 * @param[in]  offset_out     Output visibility start offset.
 * @param[out] vis            Output visibility amplitudes.
 * @param[in,out] status      Status return code.
 */
OSKAR_EXPORT
void oskar_cross_correlate(
        int source_type,
        int num_sources,
        const oskar_Jones* jones,
        const oskar_Mem* const src_flux[4],
        const oskar_Mem* const src_dir[3],
        const oskar_Mem* const src_ext[3],
        const oskar_Telescope* tel,
        const oskar_Mem* const station_uvw[3],
        double gast,
        double frequency_hz,
        int offset_out,
        oskar_Mem* vis,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
