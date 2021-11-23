/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "interferometer/private_jones.h"
#include "interferometer/oskar_jones.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_jones_join(oskar_Jones* j3, oskar_Jones* j1, const oskar_Jones* j2,
        int* status)
{
    if (*status) return;

    /* Get the dimensions of the input data. */
    if (!j3) j3 = j1;
    const int n_sources1 = j1->num_sources;
    const int n_sources2 = j2->num_sources;
    const int n_sources3 = j3->num_sources;
    const int n_stations1 = j1->num_stations;
    const int n_stations2 = j2->num_stations;
    const int n_stations3 = j3->num_stations;

    /* Check the data dimensions. */
    if (n_sources1 != n_sources2 || n_sources1 != n_sources3)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
    }
    if (n_stations1 != n_stations2 || n_stations1 != n_stations3)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
    }

    /* Multiply the array elements. */
    const size_t num_elements = n_sources1 * n_stations1;
    oskar_mem_multiply(j3->data, j1->data, j2->data,
            0, 0, 0, num_elements, status);
}

#ifdef __cplusplus
}
#endif
