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

#ifndef OSKAR_STATION_H_
#define OSKAR_STATION_H_

/**
 * @file oskar_Station.h
 */


#include "oskar_global.h"
#include "oskar_Ptr.h"

#ifdef __cplusplus
extern "C"
#endif
struct oskar_Station
{
    int num_antennas;
    oskar_Ptr antenna_x;
    oskar_Ptr antenna_y;
    oskar_Ptr antenna_z;
    oskar_Ptr antenna_weight;

    // amp and phase error.

    // Tile positions - is this the best way to do this?
    int num_tiles;
    oskar_Ptr tile_x;
    oskar_Ptr tile_y;
    oskar_Ptr tile_z;

    // Embedded element pattern.
    int num_element_patterns;
    oskar_EmbeddedElementPattern* element_pattern;

    // Station position.
    double longitude;
    double latitude;

    // Beam phase centre.
    double ra0;
    double dec0;

    int bit_depth;
};

typedef struct oskar_Station oskar_Station;


#endif // OSKAR_STATION_H_
