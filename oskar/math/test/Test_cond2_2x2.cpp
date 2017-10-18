/*
 * Copyright (c) 2015, The University of Oxford
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

#include <gtest/gtest.h>

#include "math/private_cond2_2x2.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

TEST(cond2_2x2, double_precision)
{
    double4c in;

    in.a.x = 0.8147;
    in.a.y = 0.6324;
    in.b.x = 0.1270;
    in.b.y = 0.2785;
    in.c.x = 0.9058;
    in.c.y = 0.0975;
    in.d.x = 0.9134;
    in.d.y = 0.5469;
    EXPECT_NEAR(3.5239, oskar_cond2_2x2_inline_d(&in), 1e-4);
}

TEST(cond2_2x2, single_precision)
{
    float4c in;

    in.a.x = 0.8147f;
    in.a.y = 0.6324f;
    in.b.x = 0.1270f;
    in.b.y = 0.2785f;
    in.c.x = 0.9058f;
    in.c.y = 0.0975f;
    in.d.x = 0.9134f;
    in.d.y = 0.5469f;
    EXPECT_NEAR(3.5239, oskar_cond2_2x2_inline_f(&in), 1e-4);
}
