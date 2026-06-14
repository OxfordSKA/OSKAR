/*
 * Copyright (c) 2012-2026, The OSKAR Developers.
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

/* Include enumerator lists first. */
#include "sky/oskar_sky_enum.h"

#ifdef __cplusplus
}
#endif

#include "sky/oskar_sky_accessors.h"
#include "sky/oskar_sky_append_to_set.h"
#include "sky/oskar_sky_append.h"
#include "sky/oskar_sky_check_columns.h"
#include "sky/oskar_sky_clear_source_flux.h"
#include "sky/oskar_sky_column_type_from_name.h"
#include "sky/oskar_sky_column_type_to_name.h"
#include "sky/oskar_sky_column.h"
#include "sky/oskar_sky_copy.h"
#include "sky/oskar_sky_copy_contents.h"
#include "sky/oskar_sky_create.h"
#include "sky/oskar_sky_create_columns.h"
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
#include "sky/oskar_sky_sort_columns.h"

#endif /* include guard */
