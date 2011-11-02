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
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_element_size.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_get_pointer(oskar_Mem* ref, const oskar_Mem* mem,
        const int offset, const int num_elements)
{
    size_t element_size, offset_bytes;

    if (ref == NULL || mem == 0)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (offset + num_elements > mem->private_n_elements)
        return OSKAR_ERR_OUT_OF_RANGE;

    element_size = oskar_mem_element_size(mem->private_type);
    offset_bytes = offset * element_size;
    ref->private_type = mem->private_type;
    ref->private_n_elements = num_elements;
    ref->data = (void*)((char*)mem->data + offset_bytes);
    ref->private_location = mem->private_location;
    ref->private_owner = 0;

    return 0;
}

#ifdef __cplusplus
}
#endif

