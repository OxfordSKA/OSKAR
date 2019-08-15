/*
 * Copyright (c) 2017, The University of Oxford
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

#ifndef OSKAR_IMAGER_FILTER_TIME_H_
#define OSKAR_IMAGER_FILTER_TIME_H_

/**
 * @file oskar_imager_filter_time.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Filters supplied visibility data using the time range.
 *
 * @details
 * Filters supplied visibility data using the time range,
 * if it has been set. If not set, this function returns immediately.
 *
 * @param[in,out] h             Handle to imager.
 * @param[in,out] num_vis       On input, number of supplied visibilities;
 *                              on output, number of visibilities remaining.
 * @param[in,out] uu            Baseline uu coordinates, in wavelengths.
 * @param[in,out] vv            Baseline vv coordinates, in wavelengths.
 * @param[in,out] ww            Baseline ww coordinates, in wavelengths.
 * @param[in,out] amp           Baseline complex visibility amplitudes.
 * @param[in,out] weight        Baseline visibility weights.
 * @param[in,out] time_centroid Time centroid values as MJD(UTC) _seconds_
 *                              (double precision).
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
void oskar_imager_filter_time(oskar_Imager* h, size_t* num_vis,
        oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww, oskar_Mem* amp,
        oskar_Mem* weight, oskar_Mem* time_centroid, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_IMAGER_FILTER_TIME_H_ */
