/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_H_
#define OSKAR_SKY_H_

/**
 * @file oskar_sky.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Sky;
typedef struct oskar_Sky oskar_Sky;


/* Column types. */
enum oskar_SkyColumn
{
    OSKAR_SKY_CUSTOM = 0, /* Reserved. Currently for anything unknown. */
    /* Fix numbers for original columns. */
    OSKAR_SKY_RA_RAD = 1,
    OSKAR_SKY_DEC_RAD = 2,
    OSKAR_SKY_I_JY = 3,
    OSKAR_SKY_Q_JY = 4,
    OSKAR_SKY_U_JY = 5,
    OSKAR_SKY_V_JY = 6,
    OSKAR_SKY_REF_HZ = 7,
    OSKAR_SKY_SPEC_IDX = 8,
    OSKAR_SKY_RM_RAD = 9,
    OSKAR_SKY_MAJOR_RAD = 10,
    OSKAR_SKY_MINOR_RAD = 11,
    OSKAR_SKY_PA_RAD = 12,
    OSKAR_SKY_RA_DEG,  /* Caution! Not a real column type: only for loading. */
    OSKAR_SKY_DEC_DEG, /* Caution! Not a real column type: only for loading. */
    OSKAR_SKY_LIN_SI, /* Linear spectral index (opposite to LogarithmicSI). */
    OSKAR_SKY_POLA_RAD,
    OSKAR_SKY_POLF,
    OSKAR_SKY_REF_WAVE_M,
    OSKAR_SKY_SPEC_CURV,
    OSKAR_SKY_LINE_WIDTH_HZ,
    /* Scratch columns start at 0x100 (256). */
    OSKAR_SKY_SCRATCH_START = 0x100,
    OSKAR_SKY_SCRATCH_EXT_A,
    OSKAR_SKY_SCRATCH_EXT_B,
    OSKAR_SKY_SCRATCH_EXT_C,
    OSKAR_SKY_SCRATCH_L,
    OSKAR_SKY_SCRATCH_M,
    OSKAR_SKY_SCRATCH_N,
    OSKAR_SKY_SCRATCH_I_JY,
    OSKAR_SKY_SCRATCH_Q_JY,
    OSKAR_SKY_SCRATCH_U_JY,
    OSKAR_SKY_SCRATCH_V_JY
};
typedef enum oskar_SkyColumn oskar_SkyColumn;


/* Static integer attributes. */
enum oskar_SkyAttribInt
{
    OSKAR_SKY_PRECISION,
    OSKAR_SKY_MEM_LOCATION,
    OSKAR_SKY_CAPACITY,
    OSKAR_SKY_NUM_SOURCES,
    OSKAR_SKY_NUM_COLUMNS,
    OSKAR_SKY_USE_EXTENDED,
    /* Last value is the number of attributes. */
    OSKAR_SKY_NUM_ATTRIBUTES_INT
};
typedef enum oskar_SkyAttribInt oskar_SkyAttribInt;


/* Static double attributes. */
enum oskar_SkyAttribDouble
{
    OSKAR_SKY_REF_RA_RAD,
    OSKAR_SKY_REF_DEC_RAD,
    /* Last value is the number of attributes. */
    OSKAR_SKY_NUM_ATTRIBUTES_DOUBLE
};
typedef enum oskar_SkyAttribDouble oskar_SkyAttribDouble;

#ifdef __cplusplus
}
#endif

#include "sky/oskar_sky_accessors.h"
#include "sky/oskar_sky_append_to_set.h"
#include "sky/oskar_sky_append.h"
#include "sky/oskar_sky_clear_source_flux.h"
#include "sky/oskar_sky_column_type_from_name.h"
#include "sky/oskar_sky_column_type_to_name.h"
#include "sky/oskar_sky_column.h"
#include "sky/oskar_sky_copy.h"
#include "sky/oskar_sky_copy_contents.h"
#include "sky/oskar_sky_create.h"
#include "sky/oskar_sky_create_copy.h"
#include "sky/oskar_sky_evaluate_gaussian_source_parameters.h"
#include "sky/oskar_sky_evaluate_relative_directions.h"
#include "sky/oskar_sky_filter_by_flux.h"
#include "sky/oskar_sky_filter_by_radius.h"
#include "sky/oskar_sky_free.h"
#include "sky/oskar_sky_from_fits_file.h"
#include "sky/oskar_sky_from_healpix_ring.h"
#include "sky/oskar_sky_from_image.h"
#include "sky/oskar_sky_generate_grid.h"
#include "sky/oskar_sky_generate_random_power_law.h"
#include "sky/oskar_sky_horizon_clip.h"
#include "sky/oskar_sky_load_named_columns.h"
#include "sky/oskar_sky_load.h"
#include "sky/oskar_sky_resize.h"
#include "sky/oskar_sky_save_named_columns.h"
#include "sky/oskar_sky_save.h"
#include "sky/oskar_sky_scale_flux_with_frequency.h"
#include "sky/oskar_sky_set_source.h"

#endif /* include guard */
