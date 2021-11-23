/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_set_element_weight(oskar_Station* station, int feed,
        int index, double re, double im, int* status)
{
    oskar_Mem* ptr = 0;
    if (*status || !station) return;
    ptr = station->element_weight[feed];
    if (!ptr)
    {
        station->element_weight[feed] = oskar_mem_create(
                station->precision | OSKAR_COMPLEX,
                station->mem_location, station->num_elements, status);
        ptr = station->element_weight[feed];
    }
    oskar_mem_set_element_real(ptr, 2*index + 0, re, status);
    oskar_mem_set_element_real(ptr, 2*index + 1, im, status);
}

#ifdef __cplusplus
}
#endif
