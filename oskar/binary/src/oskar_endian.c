/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include "binary/oskar_endian.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_endian(void)
{
    int num = 1;
    if (*((char*)(&num)) == 1)
        return OSKAR_LITTLE_ENDIAN;
    return OSKAR_BIG_ENDIAN;
}

void oskar_endian_swap(void* val, size_t size)
{
    char t, *d = (char*) val;
    if (size == 4)
    {
        t    = d[0];
        d[0] = d[3];
        d[3] = t;
        t    = d[1];
        d[1] = d[2];
        d[2] = t;
        return;
    }
    if (size == 8)
    {
        t    = d[0];
        d[0] = d[7];
        d[7] = t;
        t    = d[1];
        d[1] = d[6];
        d[6] = t;
        t    = d[2];
        d[2] = d[5];
        d[5] = t;
        t    = d[3];
        d[3] = d[4];
        d[4] = t;
        return;
    }
    if (size == 2)
    {
        t    = d[0];
        d[0] = d[1];
        d[1] = t;
        return;
    }
}

#ifdef __cplusplus
}
#endif
