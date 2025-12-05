/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_SKY_H_
#define OSKAR_PRIVATE_SKY_H_

/**
 * @file private_sky.h
 */

#include "mem/oskar_mem.h"
#include "sky/oskar_sky.h"

/**
 * @struct oskar_Sky
 *
 * @brief Structure to hold a sky model.
 *
 * @details
 * The structure holds source parameters for a sky model.
 */
struct oskar_Sky
{
    /* Static attributes. */
    double attr_double[OSKAR_SKY_NUM_ATTRIBUTES_DOUBLE];
    int attr_int[OSKAR_SKY_NUM_ATTRIBUTES_INT];

    /*
     * The data table needs to be in a single memory block to cater for OpenCL,
     * which does not support pointers-to-pointers (otherwise needed for a
     * variable number of columns) in kernels.
     */
    oskar_Mem* table;               /**< Data table, as a single array. */
    oskar_Mem** columns;            /**< Array of data column aliases. */
    oskar_SkyColumn* column_type;   /**< Enumerated type of each column. */
    int* column_attr;               /**< Optional column attribute. */
};

#endif /* include guard */
