/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#ifndef OSKAR_PRIVATE_STATION_WORK_H_
#define OSKAR_PRIVATE_STATION_WORK_H_

/**
 * @file private_station_work.h
 */

#include <oskar_mem.h>

#define OSKAR_MAX_STATION_DEPTH 3

/**
 * @brief
 * Structure to hold work buffers used for calculation of the station beam.
 *
 * @details
 * This structure holds work buffers used for calculation of the station beam.
 * This includes arrays to hold:
 *
 * - Horizon mask [integer].
 * - Modified source theta values [real scalar].
 * - Modified source phi values [real scalar].
 * - Beamforming weights [complex scalar].
 * - Beamforming weights error [complex scalar].
 * - Element pattern [complex matrix and complex scalar].
 * - Array pattern  [complex scalar].
 * - Hierarchy work array per beamforming level [complex matrix and complex scalar].
 *
 * Depending on the mode of operation, not all of these arrays will be used.
 */
struct oskar_StationWork
{
    oskar_Mem* horizon_mask;            /* Integer. */

    oskar_Mem* enu_direction_x;         /* Real scalar. ENU direction cosine. */
    oskar_Mem* enu_direction_y;         /* Real scalar. ENU direction cosine. */
    oskar_Mem* enu_direction_z;         /* Real scalar. ENU direction cosine. */

    oskar_Mem* theta_modified;          /* Real scalar. */
    oskar_Mem* phi_modified;            /* Real scalar. */
    oskar_Mem* weights;                 /* Complex scalar. */
    oskar_Mem* weights_error;           /* Complex scalar. */
    oskar_Mem* element_pattern_matrix;  /* Complex matrix. */
    oskar_Mem* element_pattern_scalar;  /* Complex scalar. */
    oskar_Mem* array_pattern;           /* Complex scalar. */
    oskar_Mem* beam_temp_matrix;        /* Complex matrix (normalised mode). */
    oskar_Mem* beam_temp_scalar;        /* Complex scalar (normalised mode). */

    oskar_Mem* hierarchy_work_matrix[OSKAR_MAX_STATION_DEPTH]; /* Complex matrix. */
    oskar_Mem* hierarchy_work_scalar[OSKAR_MAX_STATION_DEPTH]; /* Complex scalar. */
};

#ifndef OSKAR_STATION_WORK_TYPEDEF_
#define OSKAR_STATION_WORK_TYPEDEF_
typedef struct oskar_StationWork oskar_StationWork;
#endif /* OSKAR_STATION_WORK_TYPEDEF_ */

#endif /* OSKAR_PRIVATE_STATION_WORK_H_ */
