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


#include <mex.h>
#include <math/oskar_GridPositions.h>
#include "matlab/common/oskar_matlab_common.h"

static void error_field(const char* msg);

enum GRID_GENERATOR_TYPE
{
    CIRCULAR = 0,
    SPIRAL_ARCHIMEDEAN = 1,
    SPIRAL_LOG = 2
};

struct settings
{
    int type;
    int num_points;
    double radius;
    double radius_inner;
    double theta_start;
    double num_revs;
    double a;
};


// MATLAB Entry function.
void mexFunction(int num_out, mxArray** out, int num_in, const mxArray** in)
{
    if (num_in != 1 || num_out < 2)
    {
        oskar_matlab_usage("[x, y]", "math", "generate_grid", "<settings structure>",
                "Grid generation function.\n\n"
                "Settings structure consists of the following fields:\n"
                "[General settings]:\n"
                "  - type (enum oskar.math.type)\n"
                "  - num_points (int)\n"
                "  - radius (double)\n"
                "[Circular settings]:\n"
                "  - x spacing (double)\n"
                "  - y spacing (double)\n"
                "  - x error, perturbation std.dev. (double)\n"
                "  - y error, perturbation std.dev. (double)\n"
                "[Spiral settings]:\n"
                "  - num_revs (double)\n"
                "  - theta_start (double)\n"
                "[Archimedian Spiral settings]:\n"
                "  - a (double)\n"
                "[Log Spiral settings]:\n"
                "  - radius_inner (double)\n"
                );
    }

    // Parse input.
    if (!mxIsStruct(in[0]))
    {
        oskar_matlab_error("Invalid argument, input must be a settings structure");
    }

    settings s;

    // Read settings structure.
    mxArray* type = mxGetField(in[0], 0, "type");
    if (!type) error_field("type");
    if (!mxIsClass(type, "oskar.math.type"))
    {
        oskar_matlab_error("Invalid settings structure. The 'type' field "
                "must have a value from oskar.math.type");
    }
    mxArray* typeId_ = mxCreateNumericMatrix(1,1,mxINT32_CLASS, mxREAL);
    mexCallMATLAB(1, &typeId_, 1, &type, "uint32");
    s.type = (int)mxGetScalar(typeId_);

    // Common settings
    mxArray* radius = mxGetField(in[0], 0, "radius");
    if (!radius) error_field("radius");
    s.radius = mxGetScalar(radius);

    mxArray* num_points = mxGetField(in[0], 0, "num_points");
    if (!num_points) error_field("num_points");
    s.num_points = mxGetScalar(num_points);

    // Circular only settings.
    if (s.type == CIRCULAR)
    {
        oskar_matlab_error("Grid type unavailable");
    }
    // Common spiral settings
    if (s.type == SPIRAL_ARCHIMEDEAN || s.type == SPIRAL_LOG)
    {
        mxArray* num_revs = mxGetField(in[0], 0, "num_revs");
        if (!num_revs) error_field("num_revs");
        s.num_revs = mxGetScalar(num_revs);

        mxArray* theta_start = mxGetField(in[0], 0, "theta_start");
        if (!theta_start) error_field("theta_start");
        s.theta_start = mxGetScalar(theta_start);
    }
    if (s.type == SPIRAL_ARCHIMEDEAN)
    {
        mxArray* radius_inner = mxGetField(in[0], 0, "radius_inner");
        if (!radius_inner) error_field("radius_inner");
        s.radius_inner = mxGetScalar(radius_inner);
    }
    if (s.type == SPIRAL_LOG)
    {
        mxArray* a = mxGetField(in[0], 0, "a");
        if (!a) error_field("a");
        s.a = mxGetScalar(a);
    }

    // Create output arrays.
    out[0] = mxCreateNumericMatrix(1, s.num_points, mxDOUBLE_CLASS, mxREAL);
    out[1] = mxCreateNumericMatrix(1, s.num_points, mxDOUBLE_CLASS, mxREAL);
    double *x = (double*)mxGetData(out[0]);
    double *y = (double*)mxGetData(out[1]);

    // Run the generator.
    switch (s.type)
    {
        case CIRCULAR:
        {
            oskar_matlab_error("Grid type unavailable");
            break;
        }
        case SPIRAL_ARCHIMEDEAN:
        {
            oskar_GridPositions::spiralArchimedean<double>(s.num_points, x, y,
                    s.radius, s.radius_inner, s.num_revs, s.theta_start);
            break;
        }
        case SPIRAL_LOG:
        {
            oskar_GridPositions::spiralLog<double>(s.num_points, x, y, s.radius,
                    s.a, s.num_revs, s.theta_start);
            break;
        }
        default:
        {
            oskar_matlab_error("Unknown type");
            break;
        }
    };
}


static void error_field(const char* msg)
{
    oskar_matlab_error("Invalid settings structure, missing field: '%s'", msg);
}
