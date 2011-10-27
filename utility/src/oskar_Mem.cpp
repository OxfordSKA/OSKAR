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

#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_alloc.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_append.h"
#include "utility/oskar_mem_type_check.h"
#include <cstdlib>

oskar_Mem::oskar_Mem()
: private_type(0),
  private_location(0),
  private_n_elements(0),
  data(NULL)
{
}

oskar_Mem::oskar_Mem(int type, int location, int n_elements)
: private_type(type),
  private_location(location),
  private_n_elements(n_elements),
  data(NULL)
{
    if (n_elements > 0)
        if (oskar_mem_alloc(this) != 0)
            throw "Error in oskar_mem_alloc";
}

oskar_Mem::oskar_Mem(const oskar_Mem* other, int location)
: private_type(other->type()),
  private_location(location),
  private_n_elements(other->n_elements()),
  data(NULL)
{
    if (oskar_mem_alloc(this) != 0)
        throw "Error in oskar_mem_alloc";
    if (oskar_mem_copy(this, other) != 0) // Copy other to this.
        throw "Error in oskar_mem_copy";
}

oskar_Mem::~oskar_Mem()
{
    if (this->data != NULL)
        if (oskar_mem_free(this) != 0)
            throw "Error in oskar_mem_free";
}

int oskar_Mem::copy_to(oskar_Mem* other)
{
    return oskar_mem_copy(other, this); // Copy this to other.
}

int oskar_Mem::resize(int num_elements)
{
    return oskar_mem_realloc(this, num_elements);
}

int oskar_Mem::append(const void* from, int from_type, int from_location,
        int num_elements)
{
    return oskar_mem_append(this, from, from_type, from_location, num_elements);
}

bool oskar_Mem::is_double() const
{
    return oskar_mem_is_double(this->type());
}

bool oskar_Mem::is_complex() const
{
    return oskar_mem_is_complex(this->type());
}

bool oskar_Mem::is_scalar() const
{
    return oskar_mem_is_scalar(this->type());
}

bool oskar_Mem::is_double(const int type)
{
    return oskar_mem_is_double(type);
}

bool oskar_Mem::is_complex(const int type)
{
    return oskar_mem_is_complex(type);
}

bool oskar_Mem::is_scalar(const int type)
{
    return oskar_mem_is_scalar(type);
}
