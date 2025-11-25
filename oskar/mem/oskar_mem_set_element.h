/*
 * Copyright (c) 2014-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_SET_ELEMENT_H_
#define OSKAR_MEM_SET_ELEMENT_H_

/**
 * @file oskar_mem_set_element.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets the value of an element in a vector, either in CPU or GPU memory.
 *
 * @details
 * This function sets the value of one element in a scalar array at the
 * specified index. The array may be either in CPU or GPU memory.
 *
 * Note that the index is relative to the base precision type of the array.
 *
 * Integer types will cause an error code to be returned.
 *
 * @param[in] mem           Pointer to the block of memory to update.
 * @param[in] index         Array index to update.
 * @param[in] val           Value of element to set.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_mem_set_element_real(
        oskar_Mem* mem,
        size_t index,
        double val,
        int* status
);

/**
 * @brief
 * Sets the value of an element in a vector, either in CPU or GPU memory.
 *
 * @details
 * This function sets the value of one element in an array of pointers at the
 * specified index. The array may be either in CPU or GPU memory.
 *
 * @param[in] mem           Pointer to the block of memory to update.
 * @param[in] index         Array index to update.
 * @param[in] val           Value of element to set.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_mem_set_element_ptr(
        oskar_Mem* mem,
        size_t index,
        void* val,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
