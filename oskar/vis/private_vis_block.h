/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_VIS_BLOCK_H_
#define OSKAR_PRIVATE_VIS_BLOCK_H_

#include <mem/oskar_mem.h>

/*
 * This structure holds visibility data for all baselines, and a set of times
 * and channels.
 *
 * The dimension order is fixed. The polarisation dimension is implicit in the
 * data type (matrix or scalar) and is therefore the fastest varying.
 * From slowest to fastest varying, the remaining dimensions are:
 *
 * - Time (slowest)
 * - Channel
 * - Baseline (fastest)
 *
 * Note that it is different to that used by earlier versions of OSKAR,
 * where the order of the time and channel dimensions was swapped.
 * In addition, the Measurement Set format swaps the order of the channel
 * and baseline dimensions (so the dimension order there is
 * time, baseline, channel).
 *
 * The number of polarisations is determined by the choice of matrix or
 * scalar amplitude types. Matrix amplitude types represent 4 polarisation
 * dimensions, whereas scalar types represent Stokes-I only.
 */
struct oskar_VisBlock
{
    /* Global start time index, start channel index,
     * and maximum dimension sizes: time, channel, baseline, station. */
    int dim_start_size[6];
    int has_cross_correlations;
    int has_auto_correlations;

    /* Cross-correlation amplitude array has size:
     *     num_baselines * num_times * num_channels.
     * Autocorrelation amplitude array has size:
     *     num_stations * num_times * num_channels.
     * Polarisation dimension is implicit, and therefore fastest varying. */
    /* [complex / complex matrix] */
    oskar_Mem* cross_correlations; /* Cross-correlation visibility amplitudes. */
    oskar_Mem* auto_correlations;  /* Autocorrelation visibility amplitudes. */

    /* Baseline coordinate arrays have sizes num_baselines * num_times. */
    /* [real] */
    oskar_Mem* baseline_uvw_metres[3]; /* Baseline coordinates, in metres. */

    /* Station coordinate arrays have sizes num_stations * num_times. */
    /* [real] */
    oskar_Mem* station_uvw_metres[3]; /* Station coordinates, in metres. */
};

#ifndef OSKAR_VIS_BLOCK_TYPEDEF_
#define OSKAR_VIS_BLOCK_TYPEDEF_
typedef struct oskar_VisBlock oskar_VisBlock;
#endif /* OSKAR_VIS_BLOCK_TYPEDEF_ */

#endif /* include guard */
