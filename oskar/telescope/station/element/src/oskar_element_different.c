/*
 * Copyright (c) 2015-2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_element_different(const oskar_Element* a, const oskar_Element* b,
        int* status)
{
    int i = 0;
    if (*status) return 1;

    if (a->precision != b->precision) return 1;

    /* Check frequency-dependent data. */
    if (a->num_freq != b->num_freq) return 1;
    for (i = 0; i < b->num_freq; ++i)
    {
        if (a->freqs_hz[i] != b->freqs_hz[i]) return 1;
        if (oskar_mem_different(a->filename_x[i], b->filename_x[i], 0, status))
        {
            return 1;
        }
        if (oskar_mem_different(a->filename_y[i], b->filename_y[i], 0, status))
        {
            return 1;
        }
        if (oskar_mem_different(a->filename_scalar[i], b->filename_scalar[i],
                0, status))
        {
            return 1;
        }
    }

    /* Elements are the same. */
    return 0;
}

#ifdef __cplusplus
}
#endif
