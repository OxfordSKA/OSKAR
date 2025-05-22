/*
 * Copyright (c) 2015-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_ELEMENT_DIFFERENT_H_
#define OSKAR_ELEMENT_DIFFERENT_H_

/**
 * @file oskar_element_different.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Determines if the two element models are different.
 *
 * @details
 * This function returns true if the two supplied element models are different.
 *
 * Note that only element meta-data are checked.
 *
 * @param[in]      a        First element model.
 * @param[in]      b        Second element model.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
int oskar_element_different(const oskar_Element* a, const oskar_Element* b,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ELEMENT_DIFFERENT_H_ */
