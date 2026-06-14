/*
 * Copyright (c) 2013-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_ACCESSORS_H_
#define OSKAR_SKY_ACCESSORS_H_

/**
 * @file oskar_sky_accessors.h
 */

#include "oskar_global.h"
#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the value of the given floating-point attribute.
 *
 * @details
 * Returns the value of the given floating-point attribute.
 *
 * @param[in] sky Pointer to sky model.
 * @param[in] attribute Enumerated attribute type.
 */
OSKAR_EXPORT
double oskar_sky_double(const oskar_Sky* sky, oskar_SkyAttribDouble attribute);

/**
 * @brief Returns the value of the given integer attribute.
 *
 * @details
 * Returns the value of the given integer attribute.
 *
 * @param[in] sky Pointer to sky model.
 * @param[in] attribute Enumerated attribute type.
 */
OSKAR_EXPORT
int oskar_sky_int(const oskar_Sky* sky, oskar_SkyAttribInt attribute);

/**
 * @brief Sets the value of the given floating-point attribute.
 *
 * @details
 * Sets the value of the given floating-point attribute.
 *
 * If the attribute is read-only, this will do nothing.
 *
 * @param[in] sky Pointer to sky model.
 * @param[in] attribute Enumerated attribute type.
 * @param[in] value Value of attribute to set.
 */
OSKAR_EXPORT
void oskar_sky_set_double(
        oskar_Sky* sky,
        oskar_SkyAttribDouble attribute,
        double value
);

/**
 * @brief Sets the value of the given integer attribute.
 *
 * @details
 * Sets the value of the given integer attribute.
 *
 * If the attribute is read-only, this will do nothing.
 *
 * @param[in] sky Pointer to sky model.
 * @param[in] attribute Enumerated attribute type.
 * @param[in] value Value of attribute to set.
 */
OSKAR_EXPORT
void oskar_sky_set_int(oskar_Sky* sky, oskar_SkyAttribInt attribute, int value);

/**
 * @brief Returns the index of the first column of the specified type.
 *
 * @details
 * Returns the index of the first column of the specified type
 * in the sky model.
 *
 * @param[in] sky Pointer to sky model.
 * @param[in] column_type Enumerated column type to match.
 */
OSKAR_EXPORT
int oskar_sky_first_column(const oskar_Sky* sky, oskar_SkyColumn column_type);

/**
 * @brief Returns the number of columns of the specified type in the sky model.
 *
 * @details
 * Returns the number of columns of the specified type in the sky model.
 *
 * @param[in] sky Pointer to sky model.
 * @param[in] column_type Enumerated column type to count.
 */
OSKAR_EXPORT
int oskar_sky_num_columns_of_type(
        const oskar_Sky* sky,
        oskar_SkyColumn column_type
);

/**
 * @brief Return the number of valid columns of the given type for a source.
 *
 * @details
 * Return the number of valid columns of the given type for a source.
 *
 * @param[in] sky Pointer to sky model.
 * @param[in] column_type Enumerated column type to check.
 * @param[in] index Source index to check.
 *
 * @return An integer containing the number of valid columns of the given type
 * for the given source index.
 */
OSKAR_EXPORT
int oskar_sky_num_valid_columns_of_type(
        const oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int index
);

/**
 * @brief Set the number of valid columns of the given type for a source.
 *
 * @details
 * Set the number of valid columns of the given type for a source.
 *
 * @param[in] sky Pointer to sky model.
 * @param[in] column_type Enumerated column type to set.
 * @param[in] index Source index to set.
 * @param[in] value Number of valid columns of the specified type.
 */
OSKAR_EXPORT
void oskar_sky_set_num_valid_columns_of_type(
        oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int index,
        int value
);

/**
 * @brief Returns the attribute of the column at the given index.
 *
 * @details
 * Returns the attribute of the column at the given index.
 *
 * @param[in] sky          Pointer to sky model.
 * @param[in] column_index Index of column to check.
 */
OSKAR_EXPORT
int oskar_sky_column_attribute(const oskar_Sky* sky, int column_index);

/**
 * @brief Returns the enumerated type of the column at the given index.
 *
 * @details
 * Returns the enumerated type of the column (from oskar_SkyColumn)
 * at the given index.
 *
 * @param[in] sky          Pointer to sky model.
 * @param[in] column_index Index of column to check.
 */
OSKAR_EXPORT
oskar_SkyColumn oskar_sky_column_type(const oskar_Sky* sky, int column_index);

/**
 * @brief Sets the value of a given column at the given index.
 *
 * @details
 * Sets the value of a given column at the given index.
 *
 * @param[in] sky              Pointer to sky model.
 * @param[in] column_type      Enumerated column type.
 * @param[in] column_attribute Optional column attribute: Set to 0 if unused.
 * @param[in] index            Source index to set.
 * @param[in] value            Source parameter value to set.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_sky_set_data(
        oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int column_attribute,
        int index,
        double value,
        int* status
);

/**
 * @brief Returns the value of a given column at the given index.
 *
 * @details
 * Returns the value of a given column at the given index.
 *
 * @param[in] sky              Pointer to sky model.
 * @param[in] column_type      Enumerated column type.
 * @param[in] column_attribute Optional column attribute: Set to 0 if unused.
 * @param[in] index            Source index to return.
 */
OSKAR_EXPORT
double oskar_sky_data(
        const oskar_Sky* sky,
        oskar_SkyColumn column_type,
        int column_attribute,
        int index
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
