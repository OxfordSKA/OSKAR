/*
 * Copyright (c) 2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_CONJUGATE_H_
#define OSKAR_MEM_CONJUGATE_H_

/**
 * @file oskar_mem_conjugate.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Takes the complex conjugate of the supplied array.
 *
 * @details
 * This function takes the complex conjugate of the supplied array.
 *
 * @param[in,out] mem       Array.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_mem_conjugate(oskar_Mem* mem, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
