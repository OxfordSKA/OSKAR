/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#ifndef OSKAR_SPLINES_H_
#define OSKAR_SPLINES_H_

/**
 * @file oskar_splines.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Splines;
#ifndef OSKAR_SPLINES_TYPEDEF_
#define OSKAR_SPLINES_TYPEDEF_
typedef struct oskar_Splines oskar_Splines;
#endif /* OSKAR_SPLINES_TYPEDEF_ */

/* To maintain binary compatibility, do not change the values
 * in the lists below. */
enum OSKAR_SPLINES_TAGS
{
    OSKAR_SPLINES_TAG_NUM_KNOTS_X_THETA = 1,
    OSKAR_SPLINES_TAG_NUM_KNOTS_Y_PHI = 2,
    OSKAR_SPLINES_TAG_KNOTS_X_THETA = 3,
    OSKAR_SPLINES_TAG_KNOTS_Y_PHI = 4,
    OSKAR_SPLINES_TAG_COEFF = 5,
    OSKAR_SPLINES_TAG_SMOOTHING_FACTOR = 6
};

enum OSKAR_SPLINES_TYPE
{
    OSKAR_SPLINES_LINEAR = 0,
    OSKAR_SPLINES_SPHERICAL = 1
};

#ifdef __cplusplus
}
#endif

#include <splines/oskar_splines_accessors.h>
#include <splines/oskar_splines_copy.h>
#include <splines/oskar_splines_create.h>
#include <splines/oskar_splines_evaluate.h>
#include <splines/oskar_splines_free.h>
#include <splines/oskar_splines_fit.h>

#endif /* OSKAR_SPLINES_H_ */
