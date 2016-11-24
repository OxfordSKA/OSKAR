/*
 * Copyright (c) 2012-2016, The University of Oxford
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
#ifndef OSKAR_SKY_TYPEDEF_
#define OSKAR_SKY_TYPEDEF_
typedef struct oskar_Sky oskar_Sky;
#endif /* OSKAR_SKY_TYPEDEF_ */

/* To maintain binary compatibility, do not change the values
 * in the lists below. */
enum OSKAR_SKY_TAGS
{
    OSKAR_SKY_TAG_NUM_SOURCES = 1,
    OSKAR_SKY_TAG_DATA_TYPE = 2,
    OSKAR_SKY_TAG_RA = 3,
    OSKAR_SKY_TAG_DEC = 4,
    OSKAR_SKY_TAG_STOKES_I = 5,
    OSKAR_SKY_TAG_STOKES_Q = 6,
    OSKAR_SKY_TAG_STOKES_U = 7,
    OSKAR_SKY_TAG_STOKES_V = 8,
    OSKAR_SKY_TAG_REF_FREQ = 9,
    OSKAR_SKY_TAG_SPECTRAL_INDEX = 10,
    OSKAR_SKY_TAG_FWHM_MAJOR = 11,
    OSKAR_SKY_TAG_FWHM_MINOR = 12,
    OSKAR_SKY_TAG_POSITION_ANGLE = 13,
    OSKAR_SKY_TAG_ROTATION_MEASURE = 14
};

#ifdef __cplusplus
}
#endif

#include <sky/oskar_sky_accessors.h>
#include <sky/oskar_sky_append_to_set.h>
#include <sky/oskar_sky_append.h>
#include <sky/oskar_sky_copy.h>
#include <sky/oskar_sky_copy_contents.h>
#include <sky/oskar_sky_create.h>
#include <sky/oskar_sky_create_copy.h>
#include <sky/oskar_sky_evaluate_gaussian_source_parameters.h>
#include <sky/oskar_sky_evaluate_relative_directions.h>
#include <sky/oskar_sky_filter_by_flux.h>
#include <sky/oskar_sky_filter_by_radius.h>
#include <sky/oskar_sky_free.h>
#include <sky/oskar_sky_from_fits_file.h>
#include <sky/oskar_sky_from_healpix_ring.h>
#include <sky/oskar_sky_from_image.h>
#include <sky/oskar_sky_generate_grid.h>
#include <sky/oskar_sky_generate_random_power_law.h>
#include <sky/oskar_sky_horizon_clip.h>
#include <sky/oskar_sky_load.h>
#include <sky/oskar_sky_override_polarisation.h>
#include <sky/oskar_sky_read.h>
#include <sky/oskar_sky_resize.h>
#include <sky/oskar_sky_rotate_to_position.h>
#include <sky/oskar_sky_save.h>
#include <sky/oskar_sky_scale_flux_with_frequency.h>
#include <sky/oskar_sky_set_gaussian_parameters.h>
#include <sky/oskar_sky_set_source.h>
#include <sky/oskar_sky_set_spectral_index.h>
#include <sky/oskar_sky_write.h>


#endif /* OSKAR_SKY_H_ */
