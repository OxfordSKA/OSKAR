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

#ifndef OSKAR_WORK_STATION_BEAM_H_
#define OSKAR_WORK_STATION_BEAM_H_

/**
 * @file oskar_WorkStationBeam.h
 */

#include "oskar_global.h"
#include "utility/oskar_Mem.h"

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
 * - Element factor (G) [complex matrix].
 * - Array factor (E) [complex scalar].
 *
 * Depending on the mode of operation, not all of these arrays will be used.
 */
struct OSKAR_EXPORT oskar_WorkStationBeam
{
    oskar_Mem horizon_mask;   /* Integer. */

    oskar_Mem hor_x;          /* Real scalar. */
    oskar_Mem hor_y;          /* Real scalar. */
    oskar_Mem hor_z;          /* Real scalar. */

    oskar_Mem rel_x;          /* Real scalar */
    oskar_Mem rel_y;
    oskar_Mem rel_z;

    oskar_Mem l;              /* Real scalar. - TODO THESE dont exist yet... Tangent plane l,m,n */
    oskar_Mem m;              /* Real scalar. */
    oskar_Mem n;              /* Real scalar. */

    oskar_Mem theta_modified; /* Real scalar. */
    oskar_Mem phi_modified;   /* Real scalar. */
    oskar_Mem weights;        /* Complex scalar. */
    oskar_Mem weights_error;  /* Complex scalar. */
    oskar_Mem G_matrix;       /* Complex matrix. */
    oskar_Mem G_scalar;       /* Complex scalar. */
    oskar_Mem E;              /* Complex scalar. */

    /* TODO cuda random number states could go here...? */

#ifdef __cplusplus
    /**
     * @brief Constructor.
     *
     * @param[in] type     OSKAR memory type ID (Accepted values: OSKAR_SINGLE,
     *                     OSKAR_DOUBLE).
     * @param[in] location OSKAR memory location ID.
     */
    oskar_WorkStationBeam(int type = OSKAR_DOUBLE,
            int location = OSKAR_LOCATION_GPU);

    /**
     * @brief Destructor.
     */
    ~oskar_WorkStationBeam();
#endif
};

typedef struct oskar_WorkStationBeam oskar_WorkStationBeam;

#endif /* OSKAR_WORK_STATION_BEAM_H_ */
