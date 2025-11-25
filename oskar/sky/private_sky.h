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
    double* attr_double;            /**< Attribute values (double). */
    int* attr_int;                  /**< Attribute values (int). */
    int* column_attr;               /**< Optional column attribute. */
    oskar_SkyColumn* column_type;   /**< Enumerated type of each column. */
    oskar_Mem** columns;            /**< Array of data columns. */
    oskar_Mem* ptr_columns;         /**< Pointer to start of each column. */
};

#endif /* include guard */
