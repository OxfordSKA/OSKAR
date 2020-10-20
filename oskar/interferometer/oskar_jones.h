/*
 * Copyright (c) 2013-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_JONES_H_
#define OSKAR_JONES_H_

/**
 * @file oskar_jones.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Jones;
#ifndef OSKAR_JONES_TYPEDEF_
#define OSKAR_JONES_TYPEDEF_
typedef struct oskar_Jones oskar_Jones;
#endif /* OSKAR_JONES_TYPEDEF_ */

#ifdef __cplusplus
}
#endif

#include <interferometer/oskar_jones_accessors.h>
#include <interferometer/oskar_jones_apply_station_gains.h>
#include <interferometer/oskar_jones_create.h>
#include <interferometer/oskar_jones_create_copy.h>
#include <interferometer/oskar_jones_free.h>
#include <interferometer/oskar_jones_join.h>
#include <interferometer/oskar_jones_set_size.h>

#endif /* include guard */
