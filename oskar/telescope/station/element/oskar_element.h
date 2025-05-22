/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_ELEMENT_H_
#define OSKAR_ELEMENT_H_

/**
 * @file oskar_element.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Element;
#ifndef OSKAR_ELEMENT_TYPEDEF_
#define OSKAR_ELEMENT_TYPEDEF_
typedef struct oskar_Element oskar_Element;
#endif /* OSKAR_ELEMENT_TYPEDEF_ */

enum OSKAR_ELEMENT_TYPE
{
    OSKAR_ELEMENT_TYPE_DIPOLE,
    OSKAR_ELEMENT_TYPE_ISOTROPIC
};

enum OSKAR_ELEMENT_TAPER
{
    OSKAR_ELEMENT_TAPER_NONE,
    OSKAR_ELEMENT_TAPER_COSINE,
    OSKAR_ELEMENT_TAPER_GAUSSIAN
};

#ifdef __cplusplus
}
#endif

#include <telescope/station/element/oskar_element_accessors.h>
#include <telescope/station/element/oskar_element_copy.h>
#include <telescope/station/element/oskar_element_create.h>
#include <telescope/station/element/oskar_element_different.h>
#include <telescope/station/element/oskar_element_evaluate.h>
#include <telescope/station/element/oskar_element_free.h>
#include <telescope/station/element/oskar_element_load_spherical_wave_coeff.h>
#include <telescope/station/element/oskar_element_load_spherical_wave_coeff_feko.h>
#include <telescope/station/element/oskar_element_load_spherical_wave_coeff_galileo.h>
#include <telescope/station/element/oskar_element_resize_freq_data.h>

#endif /* include guard */
