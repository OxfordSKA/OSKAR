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
#include "utility/oskar_mem_append.h"
#include "utility/oskar_mem_append_raw.h"
#include "utility/oskar_mem_clear_contents.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_insert.h"
#include "utility/oskar_mem_load_binary.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_save_binary_append.h"
#include "utility/oskar_mem_scale_real.h"
#include "utility/oskar_mem_set_value_real.h"
#include "utility/oskar_mem_type_check.h"
#include <cstdlib>

oskar_Mem::oskar_Mem(int owner)
: private_type(0),
  private_location(0),
  private_num_elements(0),
  private_owner(owner),
  data(NULL)
{
}

oskar_Mem::oskar_Mem(int type, int location, int n_elements, int owner)
: private_type(0),
  private_location(0),
  private_num_elements(0),
  private_owner(0),
  data(NULL)
{
    if (oskar_mem_init(this, type, location, n_elements, owner))
        throw "Error in oskar_mem_init.";
}

oskar_Mem::oskar_Mem(const oskar_Mem* other, int location, int owner)
: private_type(0),
  private_location(0),
  private_num_elements(0),
  private_owner(0),
  data(NULL)
{
    if (oskar_mem_init(this, other->type(), location, other->num_elements(),
            owner))
        throw "Error in oskar_mem_init.";
    if (oskar_mem_copy(this, other)) // Copy other to this.
        throw "Error in oskar_mem_copy.";
}

oskar_Mem::~oskar_Mem()
{
    if (oskar_mem_free(this))
        throw "Error in oskar_mem_free.";
}

int oskar_Mem::append(const oskar_Mem* from)
{
    return oskar_mem_append(this, from);
}

int oskar_Mem::append_raw(const void* from, int from_type, int from_location,
        int num_elements)
{
    return oskar_mem_append_raw(this, from, from_type, from_location,
            num_elements);
}

int oskar_Mem::clear_contents()
{
    return oskar_mem_clear_contents(this);
}

int oskar_Mem::copy_to(oskar_Mem* other) const
{
    return oskar_mem_copy(other, this); // Copy this to other.
}

oskar_Mem oskar_Mem::get_pointer(int offset, int num_elements) const
{
    oskar_Mem ptr;
    if (oskar_mem_get_pointer(&ptr, this, offset, num_elements) != 0)
    {
        ptr.data = NULL;
        ptr.private_location = 0;
        ptr.private_num_elements = 0;
        ptr.private_owner = 0;
        ptr.private_type = 0;
    }
    return ptr;
}

int oskar_Mem::insert(const oskar_Mem* src, int offset)
{
    return oskar_mem_insert(this, src, offset);
}

int oskar_Mem::load_binary(const char* filename, oskar_BinaryTagIndex** index,
        const char* name_group, const char* name_tag, int user_index)
{
    return oskar_mem_load_binary(this, filename, index, name_group,
            name_tag, user_index);
}

int oskar_Mem::resize(int num_elements)
{
    return oskar_mem_realloc(this, num_elements);
}

int oskar_Mem::save_binary_append(const char* filename, const char* name_group,
        const char* name_tag, int user_index, int num_to_write) const
{
    return oskar_mem_save_binary_append(this, filename, name_group, name_tag,
            user_index, num_to_write);
}

int oskar_Mem::scale_real(double value)
{
    return oskar_mem_scale_real(this, value);
}

int oskar_Mem::set_value_real(double value)
{
    return oskar_mem_set_value_real(this, value);
}

bool oskar_Mem::is_double() const
{
    return oskar_mem_is_double(this->type());
}

bool oskar_Mem::is_single() const
{
    return oskar_mem_is_single(this->type());
}

bool oskar_Mem::is_complex() const
{
    return oskar_mem_is_complex(this->type());
}

bool oskar_Mem::is_real() const
{
    return oskar_mem_is_real(this->type());
}

bool oskar_Mem::is_scalar() const
{
    return oskar_mem_is_scalar(this->type());
}

bool oskar_Mem::is_matrix() const
{
    return oskar_mem_is_matrix(this->type());
}

// static method
bool oskar_Mem::is_double(const int type)
{
    return oskar_mem_is_double(type);
}

// static method
bool oskar_Mem::is_complex(const int type)
{
    return oskar_mem_is_complex(type);
}

// static method
bool oskar_Mem::is_scalar(const int type)
{
    return oskar_mem_is_scalar(type);
}
