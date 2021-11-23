/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_round_robin.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_round_robin(int items, int resources, int rank, int* number,
        int* start)
{
    int remainder = 0;
    *number = items / resources;
    remainder = items % resources;
    if (rank < remainder) (*number)++;
    *start = *number * rank;
    if (rank >= remainder) *start += remainder;
}

#ifdef __cplusplus
}
#endif
