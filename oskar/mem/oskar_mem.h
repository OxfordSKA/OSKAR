/*
 * Copyright (c) 2012-2017, The University of Oxford
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

#ifdef OSKAR_HAVE_OPENCL

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#endif

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
    OSKAR_GPU = 1,
    /* Bit-field, for future expansion. */
    OSKAR_CL = 2
};

#ifdef __cplusplus
}
#endif

#include <binary/oskar_binary_data_types.h>
#include <mem/oskar_mem_accessors.h>
#include <mem/oskar_mem_add.h>
#include <mem/oskar_mem_add_real.h>
#include <mem/oskar_mem_append_raw.h>
#include <mem/oskar_mem_clear_contents.h>
#include <mem/oskar_mem_copy.h>
#include <mem/oskar_mem_copy_contents.h>
#include <mem/oskar_mem_convert_precision.h>
#include <mem/oskar_mem_create.h>
#include <mem/oskar_mem_create_alias.h>
#include <mem/oskar_mem_create_alias_from_raw.h>
#include <mem/oskar_mem_create_copy.h>
#include <mem/oskar_mem_data_type_string.h>
#include <mem/oskar_mem_different.h>
#include <mem/oskar_mem_element_size.h>
#include <mem/oskar_mem_evaluate_relative_error.h>
#include <mem/oskar_mem_free.h>
#include <mem/oskar_mem_get_element.h>
#include <mem/oskar_mem_load_ascii.h>
#include <mem/oskar_mem_multiply.h>
#include <mem/oskar_mem_random_gaussian.h>
#include <mem/oskar_mem_random_range.h>
#include <mem/oskar_mem_random_uniform.h>
#include <mem/oskar_mem_read_binary_raw.h>
#include <mem/oskar_mem_read_fits_image_plane.h>
#include <mem/oskar_mem_read_healpix_fits.h>
#include <mem/oskar_mem_realloc.h>
#include <mem/oskar_mem_save_ascii.h>
#include <mem/oskar_mem_scale_real.h>
#include <mem/oskar_mem_set_alias.h>
#include <mem/oskar_mem_set_element.h>
#include <mem/oskar_mem_set_value_real.h>
#include <mem/oskar_mem_stats.h>
#include <mem/oskar_mem_write_fits_cube.h>
#include <mem/oskar_mem_write_healpix_fits.h>

#endif /* OSKAR_MEM_H_ */
