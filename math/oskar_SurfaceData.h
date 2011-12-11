/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_SURFACE_DATA_H_
#define OSKAR_SURFACE_DATA_H_

/**
 * @file oskar_SurfaceData.h
 */

#include "math/oskar_SplineData.h"
#include "utility/oskar_Mem.h"

/**
 * @brief Structure to hold surface data.
 *
 * @details
 * This structure holds the data required to describe a 2D complex surface.
 */
struct oskar_SurfaceData
{
    int num_points_x; /**< Number of points in the x direction. */
    int num_points_y; /**< Number of points in the y direction. */
    double inc_x;     /**< Increment in the x direction. */
    double inc_y;     /**< Increment in the y direction. */
    double max_x;     /**< Maximum value of x. */
    double max_y;     /**< Maximum value of y. */
    double min_x;     /**< Minimum value of x. */
    double min_y;     /**< Minimum value of y. */
    oskar_Mem re;     /**< Real part of surface (2D array). */
    oskar_Mem im;     /**< Imaginary part of surface (2D array). */

    /* The spline data are initialised at the pre-computation stage. */
    oskar_SplineData spline_re;
    oskar_SplineData spline_im;
};

typedef struct oskar_SurfaceData oskar_SurfaceData;

#endif /* OSKAR_SURFACE_DATA_H_ */
