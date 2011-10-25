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

#include "oskar_global.h"
#include "interferometry/oskar_Visibilities.h"
#include "interferometry/oskar_visibilities_append.h"
#include "interferometry/oskar_visibilities_insert.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_Mem.h"
#include <cstdio>

// Constructor.
oskar_Visibilities::oskar_Visibilities(const int num_times,
        const int num_baselines, const int num_channels, const int type,
        const int location)
: num_times(num_times),
  num_baselines(num_baselines),
  num_channels(num_channels)
{
    int num_vis = num_times * num_baselines * num_channels;

    // check if complex
    if ((type & 0x00C0) != 0x00C0)
        throw "visibilities must be complex";

    int coord_type = ((type & OSKAR_SINGLE) == OSKAR_SINGLE) ?
            OSKAR_SINGLE : OSKAR_DOUBLE;

    if (oskar_mem_init(&baseline_u, coord_type, location, num_vis))
        throw "error in oskar_mem_init";
    if (oskar_mem_init(&baseline_v, coord_type, location, num_vis))
        throw "error in oskar_mem_init";
    if (oskar_mem_init(&baseline_w, coord_type, location, num_vis))
        throw "error in oskar_mem_init";
    if (oskar_mem_init(&amplitude,  type, location, num_vis))
        throw "error in oskar_mem_init";
}

// Copy constructor.
oskar_Visibilities::oskar_Visibilities(const oskar_Visibilities* other,
        const int location)
: num_times(other->num_times),
  num_baselines(other->num_baselines),
  num_channels(other->num_channels),
  baseline_u(&other->baseline_u, location),
  baseline_v(&other->baseline_v, location),
  baseline_w(&other->baseline_w, location),
  amplitude(&other->amplitude, location)
{
}

oskar_Visibilities::~oskar_Visibilities()
{
}

int oskar_Visibilities::append(const oskar_Visibilities* other)
{
    return oskar_visibilties_append(this, other);
}

int oskar_Visibilities::insert(const oskar_Visibilities* other,
        const unsigned time_index)
{
    return oskar_visibilties_insert(this, other, time_index);
}

