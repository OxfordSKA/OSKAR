/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "interferometer/private_jones.h"

#include "interferometer/oskar_jones_accessors.h"
#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_jones_num_sources(const oskar_Jones* jones)
{
    return jones->num_sources;
}

int oskar_jones_num_stations(const oskar_Jones* jones)
{
    return jones->num_stations;
}

int oskar_jones_type(const oskar_Jones* jones)
{
    return oskar_mem_type(jones->data);
}

int oskar_jones_mem_location(const oskar_Jones* jones)
{
    return oskar_mem_location(jones->data);
}

oskar_Mem* oskar_jones_mem(oskar_Jones* jones)
{
    return jones->data;
}

const oskar_Mem* oskar_jones_mem_const(const oskar_Jones* jones)
{
    return jones->data;
}

#ifdef __cplusplus
}
#endif
