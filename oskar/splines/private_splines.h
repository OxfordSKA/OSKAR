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

#ifndef OSKAR_PRIVATE_SPLINES_H_
#define OSKAR_PRIVATE_SPLINES_H_

#include <mem/oskar_mem.h>

/*
 * This structure holds the data required to construct a surface from
 * splines.
 */
struct oskar_Splines
{
    int precision;
    int mem_location;
    int num_knots_x_theta;    /* Number of knots in x or theta. */
    int num_knots_y_phi;      /* Number of knots in y or phi. */
    oskar_Mem* knots_x_theta; /* Knot positions in x or theta. */
    oskar_Mem* knots_y_phi;   /* Knot positions in y or phi. */
    oskar_Mem* coeff;         /* Spline coefficient array. */
    double smoothing_factor;  /* Actual smoothing factor used for the fit. */
};

#ifndef OSKAR_SPLINES_TYPEDEF_
#define OSKAR_SPLINES_TYPEDEF_
typedef struct oskar_Splines oskar_Splines;
#endif /* OSKAR_SPLINES_TYPEDEF_ */

#endif /* OSKAR_PRIVATE_SPLINES_H_ */
