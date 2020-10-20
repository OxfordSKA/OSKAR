/*
 * Copyright (c) 2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_GAINS_H_
#define OSKAR_PRIVATE_GAINS_H_

#include <mem/oskar_mem.h>
#include <utility/oskar_hdf5.h>

struct oskar_Gains
{
    int precision, num_dims;
    size_t* dims;
    oskar_HDF5* hdf5_file;
    oskar_Mem* freqs;
};

#ifndef OSKAR_GAINS_TYPEDEF_
#define OSKAR_GAINS_TYPEDEF_
typedef struct oskar_Gains oskar_Gains;
#endif /* OSKAR_GAINS_TYPEDEF_ */

#endif /* include guard */
