/*
 * Copyright (c) 2022-2024, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_HARP_H_
#define OSKAR_PRIVATE_HARP_H_

#include "oskar/mem/oskar_mem.h"
#include "oskar/utility/oskar_thread.h"

struct oskar_Harp
{
    char* filename;
    int precision;
    int num_antennas;
    int num_mbf;
    int max_order;
    double freq;
    oskar_Mem *alpha_te, *alpha_tm;
    oskar_Mem *coeffs[2];
    oskar_Mem *coeffs_reordered[2];
    oskar_Mutex* mutex;
};

#ifndef OSKAR_HARP_TYPEDEF_
#define OSKAR_HARP_TYPEDEF_
typedef struct oskar_Harp oskar_Harp;
#endif /* OSKAR_HARP_TYPEDEF_ */

#endif /* include guard */
