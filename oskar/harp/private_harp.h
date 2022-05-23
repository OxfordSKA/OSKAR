/*
 * Copyright (c) 2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_HARP_H_
#define OSKAR_PRIVATE_HARP_H_

#include <mem/oskar_mem.h>
#include <utility/oskar_hdf5.h>

struct oskar_Harp
{
    int precision;
    int num_antennas;
    int num_mbf;
    int max_order;
    double freq;
    oskar_HDF5* hdf5_file;
    oskar_Mem *alpha_te, *alpha_tm;
    oskar_Mem *coeffs_pola, *coeffs_polb;
};

#ifndef OSKAR_HARP_TYPEDEF_
#define OSKAR_HARP_TYPEDEF_
typedef struct oskar_Harp oskar_Harp;
#endif /* OSKAR_HARP_TYPEDEF_ */

#endif /* include guard */
