/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif


const char* oskar_mem_data_type_string(int data_type)
{
    switch (data_type)
    {
    case OSKAR_CHAR:
        return "CHAR";
    case OSKAR_INT:
        return "INT";
    case OSKAR_SINGLE:
        return "SINGLE";
    case OSKAR_DOUBLE:
        return "DOUBLE";
    case OSKAR_COMPLEX:
        return "COMPLEX";
    case OSKAR_MATRIX:
        return "MATRIX";
    case OSKAR_SINGLE_COMPLEX:
        return "SINGLE COMPLEX";
    case OSKAR_DOUBLE_COMPLEX:
        return "DOUBLE COMPLEX";
    case OSKAR_SINGLE_COMPLEX_MATRIX:
        return "SINGLE COMPLEX MATRIX";
    case OSKAR_DOUBLE_COMPLEX_MATRIX:
        return "DOUBLE COMPLEX MATRIX";
    default:                                              /* LCOV_EXCL_LINE */
        break;                                            /* LCOV_EXCL_LINE */
    };
    return "UNKNOWN TYPE";                                /* LCOV_EXCL_LINE */
}

#ifdef __cplusplus
}
#endif
