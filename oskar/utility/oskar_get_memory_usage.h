/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_GET_MEMORY_USAGE_H_
#define OSKAR_GET_MEMORY_USAGE_H_

/**
 * @file oskar_get_memory_usage.h
 */

#include <oskar_global.h>
#include <log/oskar_log.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the total physical system memory, in bytes.
 */
OSKAR_EXPORT
size_t oskar_get_total_physical_memory(void);

/**
 * @brief Returns the free physical system memory, in bytes.
 */
OSKAR_EXPORT
size_t oskar_get_free_physical_memory(void);

/**
 * @brief Returns the memory used by the current process, in bytes.
 */
OSKAR_EXPORT
size_t oskar_get_memory_usage(void);

/**
 * @brief Writes current memory usage to the log.
 */
OSKAR_EXPORT
void oskar_log_mem(oskar_Log* log);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
