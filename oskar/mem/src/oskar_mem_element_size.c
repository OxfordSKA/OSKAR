/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif


size_t oskar_mem_element_size(int type)
{
    switch (type)
    {
    case OSKAR_CHAR:
        return sizeof(char);
    case OSKAR_INT:
        return sizeof(int);
    case OSKAR_SINGLE:
        return sizeof(float);
    case OSKAR_DOUBLE:
        return sizeof(double);
    case OSKAR_SINGLE_COMPLEX:
        return 2 * sizeof(float);
    case OSKAR_DOUBLE_COMPLEX:
        return 2 * sizeof(double);
    case OSKAR_SINGLE_COMPLEX_MATRIX:
        return 8 * sizeof(float);
    case OSKAR_DOUBLE_COMPLEX_MATRIX:
        return 8 * sizeof(double);
    default:
        return 0;
    }
}

#ifdef __cplusplus
}
#endif
