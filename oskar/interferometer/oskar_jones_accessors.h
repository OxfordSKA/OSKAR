/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_JONES_ACCESSORS_H_
#define OSKAR_JONES_ACCESSORS_H_

/**
 * @file oskar_jones_accessors.h
 */

#include <oskar_global.h>

#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Returns the number of sources in the Jones matrix block.
 *
 * @details
 * Returns the number of sources in the Jones matrix block.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return The number of stations.
 */
OSKAR_EXPORT
int oskar_jones_num_sources(const oskar_Jones* jones);

/**
 * @brief
 * Returns the number of stations in the Jones matrix block.
 *
 * @details
 * Returns the number of stations in the Jones matrix block.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return The number of stations.
 */
OSKAR_EXPORT
int oskar_jones_num_stations(const oskar_Jones* jones);

/**
 * @brief
 * Returns the enumerated data type of the Jones matrix block.
 *
 * @details
 * Returns the enumerated data type of the Jones matrix block.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return The enumerated data type.
 */
OSKAR_EXPORT
int oskar_jones_type(const oskar_Jones* jones);

/**
 * @brief
 * Returns the enumerated location of the Jones matrix block.
 *
 * @details
 * Returns the enumerated location of the Jones matrix block.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return The enumerated location.
 */
OSKAR_EXPORT
int oskar_jones_mem_location(const oskar_Jones* jones);

/**
 * @brief
 * Returns a pointer to the matrix block memory.
 *
 * @details
 * Returns a pointer to the matrix block memory.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return A pointer to the memory structure.
 */
OSKAR_EXPORT
oskar_Mem* oskar_jones_mem(oskar_Jones* jones);

/**
 * @brief
 * Returns a read-only pointer to the matrix block memory.
 *
 * @details
 * Returns a read-only pointer to the matrix block memory.
 *
 * @param[in]     jones  Pointer to data structure.
 *
 * @return A pointer to the memory structure.
 */
OSKAR_EXPORT
const oskar_Mem* oskar_jones_mem_const(const oskar_Jones* jones);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
