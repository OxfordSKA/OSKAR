/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "utility/oskar_file_exists.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_file_exists(const char* filename)
{
    FILE* stream = 0;
    if (!filename) return 0; /* Catch null pointer. */
    if (!*filename) return 0; /* Catch empty string. */
    stream = fopen(filename, "r");
    if (stream)
    {
        fclose(stream);
        return 1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif
