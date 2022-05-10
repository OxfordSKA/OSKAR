/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
#include <mem/oskar_mem_conjugate.h>
#include <mem/oskar_mem_convert_precision.h>
#include <mem/oskar_mem_create.h>
#include <mem/oskar_mem_create_alias.h>
#include <mem/oskar_mem_create_alias_from_raw.h>
#include <mem/oskar_mem_create_copy.h>
#include <mem/oskar_mem_data_type_string.h>
#include <mem/oskar_mem_different.h>
#include <mem/oskar_mem_element_size.h>
#include <mem/oskar_mem_ensure.h>
#include <mem/oskar_mem_evaluate_relative_error.h>
#include <mem/oskar_mem_free.h>
#include <mem/oskar_mem_get_element.h>
#include <mem/oskar_mem_load_ascii.h>
#include <mem/oskar_mem_multiply.h>
#include <mem/oskar_mem_normalise.h>
#include <mem/oskar_mem_random_gaussian.h>
#include <mem/oskar_mem_random_range.h>
#include <mem/oskar_mem_random_uniform.h>
#include <mem/oskar_mem_read_binary_raw.h>
#include <mem/oskar_mem_read_element.h>
#include <mem/oskar_mem_read_fits.h>
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
