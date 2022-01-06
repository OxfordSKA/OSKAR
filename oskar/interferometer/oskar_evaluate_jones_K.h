/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_EVALUATE_JONES_K_H_
#define OSKAR_EVALUATE_JONES_K_H_

/**
 * @file oskar_evaluate_jones_K.h
 */

#include <oskar_global.h>
#include <interferometer/oskar_jones.h>
#include <utility/oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the interferometer phase (K) Jones term.
 *
 * @details
 * This function constructs a set of Jones matrices that correspond to the
 * interferometer phase offset for each source and station, relative to the
 * array centre.
 *
 * The output set of Jones matrices (K) are scalar complex values.
 * This function will return an error if an incorrect type is used.
 *
 * @param[out] K                 Output set of Jones matrices.
 * @param[in]  num_sources       The number of sources in the input arrays.
 * @param[in]  l                 Source l-direction cosines.
 * @param[in]  m                 Source m-direction cosines.
 * @param[in]  n                 Source n-direction cosines.
 * @param[in]  u                 Station u coordinates, in metres.
 * @param[in]  v                 Station v coordinates, in metres.
 * @param[in]  w                 Station w coordinates, in metres.
 * @param[in]  frequency_hz      The current observing frequency, in Hz.
 * @param[in]  source_filter     Per-source values used for filtering.
 * @param[in]  source_filter_min Minimum allowed filter value (exclusive).
 * @param[in]  source_filter_max Maximum allowed filter value (inclusive).
 * @param[in]  ignore_w_components If set, ignore station w coordinate values.
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_jones_K(
        oskar_Jones* K,
        int num_sources,
        const oskar_Mem* l,
        const oskar_Mem* m,
        const oskar_Mem* n,
        const oskar_Mem* u,
        const oskar_Mem* v,
        const oskar_Mem* w,
        double frequency_hz,
        const oskar_Mem* source_filter,
        double source_filter_min,
        double source_filter_max,
        int ignore_w_components,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
