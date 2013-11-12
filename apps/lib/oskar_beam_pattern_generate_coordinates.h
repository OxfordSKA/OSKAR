/*
 * Copyright (c) 2013, The University of Oxford
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

#ifndef OSKAR_BEAM_PATTERN_GENERATE_COORDINATES_H_
#define OSKAR_BEAM_PATTERN_GENERATE_COORDINATES_H_

/**
 * @file oskar_beam_pattern_generate_coordinates.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>
#include <oskar_Settings.h>
#include <oskar_Station.h>
#include <oskar_station_work.h>
#include <oskar_SettingsBeamPattern.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Generates coordinates for use in evaluating beam patterns.
 *
 * @details
 * Coordinates are generated according to the specification in the
 * supplied settings structure and written into the appropriate field of the
 * supplied coordinate work structure.
 *
 * Note work buffer is provided to this function as initialised but unallocated.
 * It is allocated on first use.
 *
 * TODO 1) try to avoid passing station, 2) sort out settings passed to this function.
 * TODO work out what to do with the work structure.
 *
 */
OSKAR_EXPORT
void oskar_beam_pattern_generate_coordinates(oskar_Mem* x, oskar_Mem* y,
        oskar_Mem* z, int* coord_type, const oskar_SettingsBeamPattern* settings,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BEAM_PATTERN_GENERATE_COORDINATES_H_ */
