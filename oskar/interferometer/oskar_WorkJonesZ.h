/*
 * Copyright (c) 2013-2014, The University of Oxford
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


#ifndef OSKAR_WORK_JONES_Z_H_
#define OSKAR_WORK_JONES_Z_H_

/**
 * @file oskar_WorkJonesZ.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

/**
 * @brief
 * Structure to hold work buffers used for the calculation of Ionospheric
 * Jones matrices (Z Jones).
 *
 * @details
 */
struct oskar_WorkJonesZ
{
    oskar_Mem* hor_x;        /* Pierce point horizontal x coordinate */
    oskar_Mem* hor_y;        /* Pierce point horizontal y coordinate */
    oskar_Mem* hor_z;        /* Pierce point horizontal z coordinate */

    oskar_Mem* pp_lon;       /* Pierce point longitude, in radians */
    oskar_Mem* pp_lat;       /* Pierce point latitude, in radians */
    oskar_Mem* pp_rel_path;  /* Pierce point relative path length.
                                (the extra path, relative to the vertical for
                                the ionospheric column defined by the pierce
                                point) */

    oskar_Mem* screen_TEC;    /* TEC screen values for each pierce point */
    oskar_Mem* total_TEC;     /* Total TEC values for each pierce point */
};

typedef struct oskar_WorkJonesZ oskar_WorkJonesZ;


/* Utility functions -- TODO move to their own source files? */
#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
oskar_WorkJonesZ* oskar_work_jones_z_create(int type, int location, int* status);

OSKAR_EXPORT
void oskar_work_jones_z_resize(oskar_WorkJonesZ* work, int n, int* status);

OSKAR_EXPORT
void oskar_work_jones_z_free(oskar_WorkJonesZ* work, int* status);

#ifdef __cplusplus
}
#endif


#endif /* OSKAR_WORK_JONES_Z_H_ */
