/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <tables/Tables.h>

struct oskar_MeasurementSet
{
    struct MainColumns
    {
        casacore::ScalarColumn<int> antenna1, antenna2;
        casacore::ArrayColumn<float> sigma, weight;
        casacore::ScalarColumn<double> exposure, interval, time, timeCentroid;
        casacore::ArrayColumn<double> uvw;
        casacore::ArrayColumn<casacore::Complex> data;
    };
    casacore::Table* ms;  // Pointer to the Measurement Set main table.
    MainColumns msmc;     // Main table columns.
    int casa_phase_convention;
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

void oskar_ms_bind_refs(oskar_MeasurementSet* p);
