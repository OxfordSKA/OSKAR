/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_MS_CREATE_VIS_META1_H_
#define OSKAR_MS_CREATE_VIS_META1_H_

/**
 * @file oskar_ms_create_meta1.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates a blank (template) Measurement Set using the supplied meta-data.
 *
 * @details
 * This function creates an almost empty Measurement Set using only the
 * supplied meta-data. The meta-data values are written into the relevant
 * sub-tables, but the main table will be blank.
 *
 * @param[in] name The name of the output Measurement Set directory.
 * @param[in] ra The Right Ascension of the field centre in radians.
 * @param[in] dec The Declination of the field centre in radians.
 * @param[in] na The number of antennas or stations in the interferometer.
 * @param[in] ax The antenna x positions in metres, in ITRS frame (length na).
 * @param[in] ay The antenna y positions in metres, in ITRS frame (length na).
 * @param[in] az The antenna z positions in metres, in ITRS frame (length na).
 * @param[in] freq The observing frequency, in Hertz.
 */
OSKAR_MS_EXPORT
void oskar_ms_create_meta1(const char* name, double ra, double dec,
        int na, const double* ax, const double* ay, const double* az,
        double freq);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_MS_CREATE_VIS_META1_H_
