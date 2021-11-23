/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "interferometer/private_jones.h"
#include "interferometer/oskar_jones.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_jones_free(oskar_Jones* jones, int* status)
{
    if (!jones) return;
    oskar_mem_free(jones->data, status);
    free(jones);
}

#ifdef __cplusplus
}
#endif
