/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "binary/oskar_endian.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_endian(void)
{
    int num = 1;
    if (*((char*)(&num)) == 1)
    {
        return OSKAR_LITTLE_ENDIAN;
    }
    return OSKAR_BIG_ENDIAN;
}

void oskar_endian_swap(void* val, size_t size)
{
    char t = 0, *d = (char*) val;
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
