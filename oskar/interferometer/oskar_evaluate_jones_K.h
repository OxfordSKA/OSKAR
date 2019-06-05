/*
 * Copyright (c) 2011-2019, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
void oskar_evaluate_jones_K(oskar_Jones* K, int num_sources,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        const oskar_Mem* u, const oskar_Mem* v, const oskar_Mem* w,
        double frequency_hz, const oskar_Mem* source_filter,
        double source_filter_min, double source_filter_max,
        int ignore_w_components, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
