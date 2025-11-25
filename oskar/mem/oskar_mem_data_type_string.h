/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_DATA_TYPE_STRING_H_
#define OSKAR_MEM_DATA_TYPE_STRING_H_

/**
 * @file oskar_mem_data_type_string.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Returns the name of the enumerated data type.
 *
 * @details
 * Returns the name of the enumerated data type.
 *
 * @param[in] data_type Enumerated OSKAR data type.
 *
 * @return The name of the given data type.
 */
OSKAR_EXPORT
const char* oskar_mem_data_type_string(int data_type);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
