/*
 * Copyright (c) 2012-2019, The University of Oxford
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

#include <oskar_global.h>
#include <mem/oskar_mem.h>

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

OSKAR_EXPORT
int oskar_splines_precision(const oskar_Splines* data);

OSKAR_EXPORT
int oskar_splines_mem_location(const oskar_Splines* data);

OSKAR_EXPORT
int oskar_splines_have_coeffs(const oskar_Splines* data);

OSKAR_EXPORT
int oskar_splines_num_knots_x_theta(const oskar_Splines* data);

OSKAR_EXPORT
int oskar_splines_num_knots_y_phi(const oskar_Splines* data);

OSKAR_EXPORT
oskar_Mem* oskar_splines_knots_x(oskar_Splines* data);

OSKAR_EXPORT
const oskar_Mem* oskar_splines_knots_x_theta_const(const oskar_Splines* data);

OSKAR_EXPORT
oskar_Mem* oskar_splines_knots_y(oskar_Splines* data);

OSKAR_EXPORT
const oskar_Mem* oskar_splines_knots_y_phi_const(const oskar_Splines* data);

OSKAR_EXPORT
oskar_Mem* oskar_splines_coeff(oskar_Splines* data);

OSKAR_EXPORT
const oskar_Mem* oskar_splines_coeff_const(const oskar_Splines* data);

OSKAR_EXPORT
double oskar_splines_smoothing_factor(const oskar_Splines* data);

/**
 * @brief
 * Copies the contents of one data structure to another data structure.
 *
 * @details
 * This function copies data held in one structure to another structure.
 *
 * @param[out] dst          Pointer to destination data structure to copy into.
 * @param[in]  src          Pointer to source data structure to copy from.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_splines_copy(oskar_Splines* dst, const oskar_Splines* src,
        int* status);

/**
 * @brief
 * Creates and initialises a spline data structure.
 *
 * @details
 * This function creates and initialises a spline data structure,
 * and returns a handle to it.
 *
 * The data structure must be deallocated using oskar_splines_free() when it is
 * no longer required.
 *
 * @param[in] precision Enumerated type of data structure.
 * @param[in] location Enumerated location of memory held in data structure.
 * @param[in,out]  status   Status return code.
 *
 * @return A handle to the new data structure.
 */
OSKAR_EXPORT
oskar_Splines* oskar_splines_create(int precision, int location, int* status);

/**
 * @brief
 * Frees memory held by a spline data structure.
 *
 * @details
 * This function releases memory held by a spline data structure.
 *
 * @param[in,out] data Pointer to data structure.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_splines_free(oskar_Splines* data, int* status);

#ifdef __cplusplus
}
#endif

#include <splines/oskar_splines_evaluate.h>
#include <splines/oskar_splines_fit.h>

#endif /* OSKAR_SPLINES_H_ */
