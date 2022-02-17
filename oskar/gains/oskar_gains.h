/*
 * Copyright (c) 2020-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_GAINS_H_
#define OSKAR_GAINS_H_

/**
 * @file oskar_gains.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Gains;
#ifndef OSKAR_GAINS_TYPEDEF_
#define OSKAR_GAINS_TYPEDEF_
typedef struct oskar_Gains oskar_Gains;
#endif /* OSKAR_GAINS_TYPEDEF_ */

OSKAR_EXPORT
oskar_Gains* oskar_gains_create(int precision);

OSKAR_EXPORT
oskar_Gains* oskar_gains_create_copy(const oskar_Gains* other, int* status);

OSKAR_EXPORT
int oskar_gains_defined(const oskar_Gains* h);

OSKAR_EXPORT
void oskar_gains_evaluate(const oskar_Gains* h, int time_index_sim,
        double frequency_hz, oskar_Mem* gains, int feed, int* status);

OSKAR_EXPORT
void oskar_gains_free(oskar_Gains* h, int* status);

OSKAR_EXPORT
void oskar_gains_open_hdf5(oskar_Gains* h, const char* path, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
