/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_ADD_INLINE_H_
#define OSKAR_ADD_INLINE_H_

/**
 * @file oskar_add_inline.h
 */

#include <oskar_global.h>
#ifdef __CUDACC__
/* Must include this first to avoid type conflicts. */
#include <vector_types.h>
#endif
#include <oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Adds two complex numbers (single precision).
 *
 * @details
 * This function adds two complex numbers.
 *
 * @param[in,out] a The accumulated complex number.
 * @param[in] b     The complex number to add.
 */
OSKAR_INLINE
void oskar_add_complex_in_place_f(float2* a, const float2* b)
{
    a->x += b->x;
    a->y += b->y;
}

/**
 * @brief
 * Adds two complex numbers (double precision).
 *
 * @details
 * This function adds two complex numbers.
 *
 * @param[in,out] a The accumulated complex number.
 * @param[in] b     The complex number to add.
 */
OSKAR_INLINE
void oskar_add_complex_in_place_d(double2* a, const double2* b)
{
    a->x += b->x;
    a->y += b->y;
}

/**
 * @brief
 * Adds two complex matrices (single precision).
 *
 * @details
 * This function adds two complex matrices.
 *
 * @param[in,out] a The accumulated complex matrix.
 * @param[in] b     The complex matrix to add.
 */
OSKAR_INLINE
void oskar_add_complex_matrix_in_place_f(float4c* a, const float4c* b)
{
    a->a.x += b->a.x;
    a->a.y += b->a.y;
    a->b.x += b->b.x;
    a->b.y += b->b.y;
    a->c.x += b->c.x;
    a->c.y += b->c.y;
    a->d.x += b->d.x;
    a->d.y += b->d.y;
}

/**
 * @brief
 * Adds two complex matrices (double precision).
 *
 * @details
 * This function adds two complex matrices.
 *
 * @param[in,out] a The accumulated complex matrix.
 * @param[in] b     The complex matrix to add.
 */
OSKAR_INLINE
void oskar_add_complex_matrix_in_place_d(double4c* a, const double4c* b)
{
    a->a.x += b->a.x;
    a->a.y += b->a.y;
    a->b.x += b->b.x;
    a->b.y += b->b.y;
    a->c.x += b->c.x;
    a->c.y += b->c.y;
    a->d.x += b->d.x;
    a->d.y += b->d.y;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ADD_INLINE_H_ */
