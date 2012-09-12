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

#include "utility/oskar_mem_all_headers.h"
#include <cstdlib>

oskar_Mem::oskar_Mem(int owner_)
{
    oskar_mem_init(this, 0, 0, 0, owner_);
}

oskar_Mem::oskar_Mem(int mem_type, int mem_location, int size, int owner_)
{
    int err = oskar_mem_init(this, mem_type, mem_location, size, owner_);
    if (err) throw err;
}

oskar_Mem::oskar_Mem(const oskar_Mem* other, int mem_location, int owner_)
{
    int err;
    err = oskar_mem_init(this, other->type, mem_location, other->num_elements,
            owner_);
    oskar_mem_copy(this, other, &err); // Copy other to this.
    if (err) throw err;
}

oskar_Mem::~oskar_Mem()
{
    int err = oskar_mem_free(this);
    if (err) throw err;
}

int oskar_Mem::scale_real(double value)
{
    return oskar_mem_scale_real(this, value);
}

int oskar_Mem::set_value_real(double value)
{
    return oskar_mem_set_value_real(this, value);
}

int oskar_Mem::is_double() const
{
    return oskar_mem_is_double(this->type);
}

int oskar_Mem::is_single() const
{
    return oskar_mem_is_single(this->type);
}

int oskar_Mem::is_complex() const
{
    return oskar_mem_is_complex(this->type);
}

int oskar_Mem::is_real() const
{
    return oskar_mem_is_real(this->type);
}

int oskar_Mem::is_scalar() const
{
    return oskar_mem_is_scalar(this->type);
}

int oskar_Mem::is_matrix() const
{
    return oskar_mem_is_matrix(this->type);
}

// static method
int oskar_Mem::is_double(const int type_)
{
    return oskar_mem_is_double(type_);
}

// static method
int oskar_Mem::is_complex(const int type_)
{
    return oskar_mem_is_complex(type_);
}

// static method
int oskar_Mem::is_scalar(const int type_)
{
    return oskar_mem_is_scalar(type_);
}
