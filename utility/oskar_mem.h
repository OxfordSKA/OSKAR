/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#ifndef OSKAR_MEM_H_
#define OSKAR_MEM_H_

/**
 * @file oskar_mem.h
 *
 * @brief Structure to wrap a memory pointer either on the CPU or GPU.
 *
 * @details
 * This structure holds a pointer to memory either on the CPU
 * or GPU, and defines the type of the data to which it points.
 *
 * The structure will normally take ownership of the memory, so the
 * memory will be released when the structure is freed.
 */

#include <oskar_global.h>

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Mem;
#ifndef OSKAR_MEM_TYPEDEF_
#define OSKAR_MEM_TYPEDEF_
typedef struct oskar_Mem oskar_Mem;
#endif /* OSKAR_MEM_TYPEDEF_ */

enum OSKAR_MEM_LOCATION
{
    OSKAR_CPU = 0,
    OSKAR_GPU = 1
};

#ifdef __cplusplus
}
#endif

#include <oskar_binary_data_types.h>
#include <oskar_mem_accessors.h>
#include <oskar_mem_add.h>
#include <oskar_mem_append_raw.h>
#include <oskar_mem_clear_contents.h>
#include <oskar_mem_copy.h>
#include <oskar_mem_copy_contents.h>
#include <oskar_mem_convert_precision.h>
#include <oskar_mem_create.h>
#include <oskar_mem_create_alias.h>
#include <oskar_mem_create_alias_from_raw.h>
#include <oskar_mem_create_copy.h>
#include <oskar_mem_data_type_string.h>
#include <oskar_mem_different.h>
#include <oskar_mem_element_multiply.h>
#include <oskar_mem_element_size.h>
#include <oskar_mem_evaluate_relative_error.h>
#include <oskar_mem_free.h>
#include <oskar_mem_get_element.h>
#include <oskar_mem_load_ascii.h>
#include <oskar_mem_random_gaussian.h>
#include <oskar_mem_random_range.h>
#include <oskar_mem_random_uniform.h>
#include <oskar_mem_read_binary_raw.h>
#include <oskar_mem_realloc.h>
#include <oskar_mem_save_ascii.h>
#include <oskar_mem_scale_real.h>
#include <oskar_mem_set_alias.h>
#include <oskar_mem_set_element.h>
#include <oskar_mem_set_value_real.h>
#include <oskar_mem_stats.h>

#endif /* OSKAR_MEM_H_ */
