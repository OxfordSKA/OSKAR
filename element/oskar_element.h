/*
 * Copyright (c) 2013-2014, The University of Oxford
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

/* To maintain binary compatibility, do not change the values
 * in the lists below. */
enum OSKAR_ELEMENT_TAGS
{
    OSKAR_ELEMENT_TAG_SURFACE_TYPE = 1
};

enum OSKAR_ELEMENT_SURFACE_TYPE
{
    OSKAR_ELEMENT_SURFACE_TYPE_SCALAR = 0,
    OSKAR_ELEMENT_SURFACE_TYPE_LUDWIG_3 = 1
};

enum OSKAR_ELEMENT_TYPE
{
    OSKAR_ELEMENT_TYPE_DIPOLE,
    OSKAR_ELEMENT_TYPE_GEOMETRIC_DIPOLE,
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

#include <oskar_element_accessors.h>
#include <oskar_element_copy.h>
#include <oskar_element_create.h>
#include <oskar_element_evaluate.h>
#include <oskar_element_free.h>
#include <oskar_element_load_cst.h>
#include <oskar_element_load_scalar.h>
#include <oskar_element_resize_freq_data.h>
#include <oskar_element_read.h>
#include <oskar_element_write.h>

#endif /* OSKAR_ELEMENT_H_ */
