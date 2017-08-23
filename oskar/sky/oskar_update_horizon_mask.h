/*
 * Copyright (c) 2013-2017, The University of Oxford
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

#ifndef OSKAR_UPDATE_HORIZON_MASK_H_
#define OSKAR_UPDATE_HORIZON_MASK_H_

/**
 * @file oskar_update_horizon_mask.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Ensures source mask value is 1 if the source is visible.
 *
 * @details
 * This function sets the horizon mask to 1 if a source is
 * visible from a particular station.
 *
 * @param[in] num_sources The number of source positions.
 * @param[in] l           Source l-direction cosines relative to phase centre.
 * @param[in] m           Source m-direction cosines relative to phase centre.
 * @param[in] n           Source n-direction cosines relative to phase centre.
 * @param[in] ha0_rad     Local apparent hour angle of phase centre, in radians.
 * @param[in] dec0_rad    Local apparent declination of phase centre, in radians.
 * @param[in] lat_rad     The observer's geodetic latitude, in radians.
 * @param[in,out] mask    The input and output mask vector.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_update_horizon_mask(int num_sources, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, const double ha0_rad,
        const double dec0_rad, const double lat_rad, oskar_Mem* mask,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_UPDATE_HORIZON_MASK_H_ */
