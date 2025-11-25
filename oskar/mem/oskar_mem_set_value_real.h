/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_SET_VALUE_REAL_H_
#define OSKAR_MEM_SET_VALUE_REAL_H_

/**
 * @file oskar_mem_set_value_real.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets the value of all elements in a vector.
 *
 * @details
 * This function sets all the values in a block of memory to the same, real,
 * value. For complex types, the imaginary components are set to zero, and
 * for matrix types, the off-diagonal elements are set to zero.
 *
 * Note that a value of zero for both the \p offset and \p length parameters
 * will cause the entire array to be set.
 *
 * Integer types will cause an error code to be returned.
 *
 * @param[in,out] mem          The block of memory to update.
 * @param[in]     value        Elements will be set to this value.
 * @param[in]     offset       Array index offset at which to start.
 * @param[in]     num_elements Number of array elements to set.
 *                             Note that 0 for both \p offset
 *                             and \p num_elements means "all".
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_mem_set_value_real(
        oskar_Mem* mem,
        double value,
        size_t offset,
        size_t num_elements,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
