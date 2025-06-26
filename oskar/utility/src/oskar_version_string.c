/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "utility/oskar_version_string.h"
#include "oskar_version.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* oskar_version_string(void)
{
    return OSKAR_VERSION_STR;
}

#ifdef __cplusplus
}
#endif
