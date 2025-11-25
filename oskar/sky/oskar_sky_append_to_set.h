/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_APPEND_TO_SET_H_
#define OSKAR_SKY_APPEND_TO_SET_H_

/**
 * @file oskar_sky_append_to_set.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Append a sky model into an array of sky models of fixed number of
 * sources.
 *
 * @details
 * This function is used to assemble a set of sky models (sky chunks) used
 * for the interferometry simulation.
 *
 * @param[in,out] set_size          Updated number of sky models in the set.
 * @param[in,out] set               Pointer to array of sky model handles.
 * @param[in] max_sources_per_model Maximum number of sources per sky model.
 * @param[in] model                 Sky model to add to the set.
 * @param[in,out] status            Status return code.
 */
OSKAR_EXPORT
void oskar_sky_append_to_set(
        int* set_size,
        oskar_Sky*** set,
        int max_sources_per_model,
        const oskar_Sky* model,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
