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


#ifndef OSKAR_EVALUATE_SOURCE_HORIZONTAL_LMN_H_
#define OSKAR_EVALUATE_SOURCE_HORIZONTAL_LMN_H_

/**
 * @file oskar_evaluate_source_horizontal_lmn.h
 */

#include "oskar_global.h"
#include "sky/oskar_SkyModel.h"
#include "station/oskar_StationModel.h"
#include "station/oskar_WorkE.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Wrapper function to evaluate the horizontal direction cosines
 * for all sources in the specified sky model for the specified Greenwich
 * apparent sidereal time.
 *
 * @details
 * Converts source positions in the sky model from equatorial (RA, Dec)
 * coordinates to horizontal lmn coordinates at the specified time for the
 * station position specified by the station model longitude and latitude.
 *
 * @param[out] work    OSKAR E Jones work buffer structure containing arrays
 *                     to hold the hoizontal lmn coordinates of each source.
 * @param[in]  sky     Sky Model.
 * @param[in]  station Station Model.
 * @param[in]  gast    The Greenwich apparent sidereal time, in radians.
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_evaluate_source_horizontal_lmn(oskar_WorkE* work,
        const oskar_SkyModel* sky, const oskar_StationModel* station,
        const double gast);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_SOURCE_HORIZONTAL_LMN_H_ */
