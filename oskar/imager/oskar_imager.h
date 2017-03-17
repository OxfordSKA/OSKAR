/*
 * Copyright (c) 2016, The University of Oxford
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

#ifndef OSKAR_IMAGER_H_
#define OSKAR_IMAGER_H_

/**
 * @file oskar_imager.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Imager;
#ifndef OSKAR_IMAGER_TYPEDEF_
#define OSKAR_IMAGER_TYPEDEF_
typedef struct oskar_Imager oskar_Imager;
#endif /* OSKAR_IMAGER_TYPEDEF_ */

enum OSKAR_IMAGE_TYPE
{
    OSKAR_IMAGE_TYPE_STOKES, /* IQUV */
    OSKAR_IMAGE_TYPE_I,
    OSKAR_IMAGE_TYPE_Q,
    OSKAR_IMAGE_TYPE_U,
    OSKAR_IMAGE_TYPE_V,
    OSKAR_IMAGE_TYPE_LINEAR, /* all linear polarisations XX,XY,YX,YY */
    OSKAR_IMAGE_TYPE_XX,
    OSKAR_IMAGE_TYPE_YY,
    OSKAR_IMAGE_TYPE_XY,
    OSKAR_IMAGE_TYPE_YX,
    OSKAR_IMAGE_TYPE_PSF
};

enum OSKAR_IMAGE_ALGORITHM
{
    OSKAR_ALGORITHM_FFT,
    OSKAR_ALGORITHM_DFT_2D,
    OSKAR_ALGORITHM_DFT_3D,
    OSKAR_ALGORITHM_WPROJ,
    OSKAR_ALGORITHM_AWPROJ
};

enum OSKAR_IMAGE_WEIGHTING
{
    OSKAR_WEIGHTING_NATURAL,
    OSKAR_WEIGHTING_RADIAL,
    OSKAR_WEIGHTING_UNIFORM,
    OSKAR_WEIGHTING_GRIDLESS_UNIFORM
};

#ifdef __cplusplus
}
#endif

#include <imager/oskar_imager_accessors.h>
#include <imager/oskar_imager_check_init.h>
#include <imager/oskar_imager_create.h>
#include <imager/oskar_imager_finalise.h>
#include <imager/oskar_imager_free.h>
#include <imager/oskar_imager_linear_to_stokes.h>
#include <imager/oskar_imager_reset_cache.h>
#include <imager/oskar_imager_rotate_coords.h>
#include <imager/oskar_imager_rotate_vis.h>
#include <imager/oskar_imager_run.h>
#include <imager/oskar_imager_update.h>

#endif /* OSKAR_IMAGER_H_ */
