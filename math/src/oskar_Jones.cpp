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

#include "math/oskar_Jones.h"
#include "math/oskar_jones_join.h"
#include "math/oskar_jones_set_real_scalar.h"
#include "utility/oskar_mem_copy.h"
#include <cstdlib>

oskar_Jones::oskar_Jones(int type, int location, int num_stations,
		int num_sources)
: private_num_stations(num_stations),
  private_num_sources(num_sources),
  ptr(type, location, num_sources * num_stations)
{
}

oskar_Jones::oskar_Jones(const oskar_Jones* other, int location)
: private_num_stations(other->num_stations()),
  private_num_sources(other->num_sources()),
  ptr(&other->ptr, location)
{
}

oskar_Jones::~oskar_Jones()
{
}

int oskar_Jones::copy_to(oskar_Jones* other)
{
    return oskar_mem_copy(&other->ptr, &ptr); // Copy this to other.
}

int oskar_Jones::join_from_right(const oskar_Jones* other)
{
    return oskar_jones_join(NULL, this, other);
}

int oskar_Jones::join_to_left(oskar_Jones* other) const
{
    return oskar_jones_join(NULL, other, this);
}

int oskar_Jones::set_real_scalar(double scalar)
{
    return oskar_jones_set_real_scalar(this, scalar);
}
