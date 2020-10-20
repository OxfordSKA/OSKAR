/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <ms/MeasurementSets.h>

struct oskar_MeasurementSet
{
    casacore::MeasurementSet* ms;   // Pointer to the Measurement Set.
    casacore::MSColumns* msc;       // Pointer to the sub-tables.
    casacore::MSMainColumns* msmc;  // Pointer to the main columns.
    char* app_name;
    unsigned int *a1, *a2;
    unsigned int num_pols, num_channels, num_stations, num_receptors;
    int data_written;
    int phase_centre_type;
    double phase_centre_rad[2];
    double freq_start_hz, freq_inc_hz;
    double start_time, end_time, time_inc_sec;
};
#ifndef OSKAR_MEASUREMENT_SET_TYPEDEF_
#define OSKAR_MEASUREMENT_SET_TYPEDEF_
typedef struct oskar_MeasurementSet oskar_MeasurementSet;
#endif /* OSKAR_MEASUREMENT_SET_TYPEDEF_ */

