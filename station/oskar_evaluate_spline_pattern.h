/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_EVALUATE_SPLINE_PATTERN_H_
#define OSKAR_EVALUATE_SPLINE_PATTERN_H_

/**
 * @file oskar_evaluate_spline_pattern.h
 */

#include "oskar_global.h"
#include "station/oskar_ElementModel.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_Work.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates patterns of two perfect dipoles at source positions for given
 * orientations.
 *
 * @details
 * This function evaluates the patterns of two perfect dipole antennas
 * at the supplied source positions.
 *
 * The dipole orientation angles specify the dipole axis as the angle
 * East (x) from North (y).
 *
 * The output matrix is
 *
 * ( g_theta^a   g_phi^a )
 * ( g_theta^b   g_phi^b )
 *
 * where phi and theta are the angles measured from x to y and from xy to z,
 * respectively.
 *
 * The 'a' dipole is nominally along the x axis, and
 * the 'b' dipole is nominally along the y axis.
 * The azimuth orientation of 'a' should normally be 90 degrees, and
 * the azimuth orientation of 'b' should normally be 0 degrees.
 *
 * <b>NOTE that the dipole orientations are ignored in the current version.</b>
 *
 * @param[out] pattern           Array of output Jones matrices per source.
 * @param[in] element            Pointer to element model containing spline data.
 * @param[in] l                  Source direction cosines in x.
 * @param[in] m                  Source direction cosines in y.
 * @param[in] n                  Source direction cosines in z.
 * @param[in] cos_orientation_x  The cosine of the azimuth angle of nominal x dipole.
 * @param[in] sin_orientation_x  The sine of the azimuth angle of nominal x dipole.
 * @param[in] cos_orientation_y  The cosine of the azimuth angle of nominal y dipole.
 * @param[in] sin_orientation_y  The sine of the azimuth angle of nominal y dipole.
 * @param[in] work               Pointer to work buffer.
 */
OSKAR_EXPORT
int oskar_evaluate_spline_pattern(oskar_Mem* pattern,
        const oskar_ElementModel* element, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, double cos_orientation_x,
        double sin_orientation_x, double cos_orientation_y,
        double sin_orientation_y, oskar_Work* work);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_SPLINE_PATTERN_H_ */
