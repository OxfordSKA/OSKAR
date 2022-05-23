/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_MEM_FREE_H_
#define OSKAR_MEM_FREE_H_

/**
 * @file oskar_mem_free.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Decrements the reference counter.
 *
 * @details
 * Decrements the reference counter.
 *
 * If the reference counter reaches zero, any memory owned by the handle
 * will be released, and the handle destroyed.
 *
 * @param[in] mem Pointer to data structure whose memory to free.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_mem_free(oskar_Mem* mem, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
