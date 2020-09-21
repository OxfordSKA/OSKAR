/*
 * Copyright (c) 2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_HDF5_H_
#define OSKAR_PRIVATE_HDF5_H_

#include <stdint.h>

struct oskar_HDF5
{
    int64_t file_id;
    int num_datasets;
    char** names;
};
#ifndef OSKAR_HDF5_TYPEDEF_
#define OSKAR_HDF5_TYPEDEF_
typedef struct oskar_HDF5 oskar_HDF5;
#endif

#endif /* include guard */
