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

#ifndef OSKAR_IMAGE_H_
#define OSKAR_IMAGE_H_

/**
 * @file oskar_image.h
 *
 * @deprecated
 * The oskar_Image structure is deprecated.
 * Do not use these functions or enumerators in new code.
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Image;
#ifndef OSKAR_IMAGE_TYPEDEF_
#define OSKAR_IMAGE_TYPEDEF_
typedef struct oskar_Image oskar_Image;
#endif /* OSKAR_IMAGE_TYPEDEF_ */

enum OSKAR_IMAGE_TYPE
{
    OSKAR_IMAGE_TYPE_UNDEF = 0,

    OSKAR_IMAGE_TYPE_STOKES   = 1, /* IQUV */
    OSKAR_IMAGE_TYPE_STOKES_I = 2,
    OSKAR_IMAGE_TYPE_STOKES_Q = 3,
    OSKAR_IMAGE_TYPE_STOKES_U = 4,
    OSKAR_IMAGE_TYPE_STOKES_V = 5,

    OSKAR_IMAGE_TYPE_POL_LINEAR = 6, /* all linear polarisations XX,XY,YX,YY */
    OSKAR_IMAGE_TYPE_POL_XX     = 7,
    OSKAR_IMAGE_TYPE_POL_YY     = 8,
    OSKAR_IMAGE_TYPE_POL_XY     = 9,
    OSKAR_IMAGE_TYPE_POL_YX     = 10,

    OSKAR_IMAGE_TYPE_PSF        = 50,

    OSKAR_IMAGE_TYPE_BEAM_SCALAR = 100,
    OSKAR_IMAGE_TYPE_BEAM_POLARISED = 101
};

#ifdef __cplusplus
}
#endif

#include <oskar_image_accessors.h>
#include <oskar_image_create.h>
#include <oskar_image_free.h>
#include <oskar_image_resize.h>

#endif /* OSKAR_IMAGE_H_ */
