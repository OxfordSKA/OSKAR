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

#include "interferometry/oskar_visibilities_all_headers.h"

oskar_Visibilities::oskar_Visibilities(int amp_type, int location,
        int num_channels, int num_times, int num_stations)
{
    int err = 0;
    oskar_visibilities_init(this, amp_type, location, num_channels,
            num_times, num_stations, &err);
    if (err) throw err;
}

oskar_Visibilities::oskar_Visibilities(const oskar_Visibilities* other,
        int location)
{
    int err = 0;
    oskar_visibilities_init(this, other->amplitude.type, location,
            other->num_channels, other->num_times, other->num_stations, &err);
    oskar_visibilities_copy(this, other, &err); // Copy other to this.
    if (err) throw err;
}

oskar_Visibilities::~oskar_Visibilities()
{
    int err = oskar_visibilities_free(this);
    if (err) throw err;
}

int oskar_Visibilities::write(oskar_Log* log, const char* filename)
{
    return oskar_visibilities_write(this, log, filename);
}

int oskar_Visibilities::read(oskar_Visibilities* vis, const char* filename)
{
    return oskar_visibilities_read(vis, filename);
}

int oskar_Visibilities::get_channel_amps(oskar_Mem* vis_amps, int channel)
{
    return oskar_visibilties_get_channel_amps(vis_amps, this, channel);
}

int oskar_Visibilities::location() const
{
    return oskar_visibilities_location(this);
}
