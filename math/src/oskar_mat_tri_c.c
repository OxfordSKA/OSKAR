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

#include "math/oskar_mat_tri_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_mat_tri_c_f(int n, const float* a, float* b)
{
    int a1, a2, p, q = 0;
    for (a1 = 0; a1 < n; ++a1)
    {
        for (a2 = a1 + 1; a2 < n; ++a2, q += 2)
        {
            p = 2 * (a2 + a1 * n);
            b[q]     = a[p];
            b[q + 1] = a[p + 1];
        }
    }
}

/* Double precision. */
void oskar_mat_tri_c_d(int n, const double* a, double* b)
{
    int a1, a2, p, q = 0;
    for (a1 = 0; a1 < n; ++a1)
    {
        for (a2 = a1 + 1; a2 < n; ++a2, q += 2)
        {
            p = 2 * (a2 + a1 * n);
            b[q]     = a[p];
            b[q + 1] = a[p + 1];
        }
    }
}

#ifdef __cplusplus
}
#endif
