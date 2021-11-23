/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_different(const oskar_Mem* one, const oskar_Mem* two,
        size_t num_elements, int* status)
{
    size_t bytes_to_check = 0;

    /* Check if safe to proceed. */
    if (*status) return OSKAR_TRUE;

    /* Check that both arrays exist. */
    if ((!one && two) || (one && !two)) return OSKAR_TRUE;

    /* If neither array exists, return false. */
    if (!one && !two) return OSKAR_FALSE;

    /* Check the data types. */
    if (one->type != two->type) return OSKAR_TRUE;

    /* Check the number of elements. */
    if (num_elements == 0 || num_elements > one->num_elements)
    {
        num_elements = one->num_elements;
    }
    if (num_elements > two->num_elements)
    {
        return OSKAR_TRUE;
    }
    bytes_to_check = num_elements * oskar_mem_element_size(one->type);

    /* Check data location. */
    if (one->location == OSKAR_CPU && two->location == OSKAR_CPU)
    {
        return (memcmp(one->data, two->data, bytes_to_check) != 0);
    }

    /* Data checks are only supported in CPU memory. */
    *status = OSKAR_ERR_BAD_LOCATION;
    return OSKAR_TRUE;
}

#ifdef __cplusplus
}
#endif
