/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "ms/oskar_measurement_set.h"
#include "ms/private_ms.h"

#include <tables/Tables.h>
#include <casa/Arrays/Vector.h>

using namespace casacore;

static void oskar_ms_create_baseline_indices(oskar_MeasurementSet* p,
        unsigned int num_baselines)
{
    bool write_auto_corr = false, write_cross_corr = false;
    unsigned int num_stations = p->num_stations;
    size_t size_bytes = num_baselines * sizeof(unsigned int);
    p->a1 = (unsigned int*) realloc(p->a1, size_bytes);
    p->a2 = (unsigned int*) realloc(p->a2, size_bytes);
    if (num_baselines == num_stations * (num_stations + 1) / 2)
    {
        write_auto_corr = true;
        write_cross_corr = true;
    }
    else if (num_baselines == num_stations * (num_stations - 1) / 2)
    {
        write_auto_corr = false;
        write_cross_corr = true;
    }
    else if (num_baselines == num_stations)
    {
        write_auto_corr = true;
        write_cross_corr = false;
    }
    if (write_cross_corr || write_auto_corr)
    {
        for (unsigned int s1 = 0, i = 0; s1 < num_stations; ++s1)
        {
            if (write_auto_corr)
            {
                p->a1[i] = s1;
                p->a2[i] = s1;
                ++i;
            }
            if (write_cross_corr)
            {
                for (unsigned int s2 = s1 + 1; s2 < num_stations; ++i, ++s2)
                {
                    p->a1[i] = s1;
                    p->a2[i] = s2;
                }
            }
        }
    }
    else
    {
        memset(p->a1, 0, num_baselines * sizeof(int));
        memset(p->a2, 0, num_baselines * sizeof(int));
    }
}

template <typename T>
void oskar_ms_write_coords(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int num_baselines,
        const T* uu, const T* vv, const T* ww,
        double exposure_sec, double interval_sec, double time_stamp)
{
    // Allocate storage for a (u,v,w) coordinate and a visibility weight.
    Vector<Double> uvw(3);
    Vector<Float> weight(p->num_pols, 1.0), sigma(p->num_pols, 1.0);

    // Get references to columns.
#ifdef OSKAR_MS_NEW
    ArrayColumn<Double>& col_uvw = p->msmc.uvw;
    ScalarColumn<Int>& col_antenna1 = p->msmc.antenna1;
    ScalarColumn<Int>& col_antenna2 = p->msmc.antenna2;
    ArrayColumn<Float>& col_weight = p->msmc.weight;
    ArrayColumn<Float>& col_sigma = p->msmc.sigma;
    ScalarColumn<Double>& col_exposure = p->msmc.exposure;
    ScalarColumn<Double>& col_interval = p->msmc.interval;
    ScalarColumn<Double>& col_time = p->msmc.time;
    ScalarColumn<Double>& col_timeCentroid = p->msmc.timeCentroid;
#else
    MSMainColumns* msmc = p->msmc;
    if (!msmc) return;
    ArrayColumn<Double>& col_uvw = msmc->uvw();
    ScalarColumn<Int>& col_antenna1 = msmc->antenna1();
    ScalarColumn<Int>& col_antenna2 = msmc->antenna2();
    ArrayColumn<Float>& col_weight = msmc->weight();
    ArrayColumn<Float>& col_sigma = msmc->sigma();
    ScalarColumn<Double>& col_exposure = msmc->exposure();
    ScalarColumn<Double>& col_interval = msmc->interval();
    ScalarColumn<Double>& col_time = msmc->time();
    ScalarColumn<Double>& col_timeCentroid = msmc->timeCentroid();
#endif

    // Add new rows if required.
    oskar_ms_ensure_num_rows(p, start_row + num_baselines);

    // Create baseline antenna indices if required.
    if (!p->a1 || !p->a2)
    {
        oskar_ms_create_baseline_indices(p, num_baselines);
    }

    // Loop over rows to add.
    for (unsigned int r = 0; r < num_baselines; ++r)
    {
        // Write the data to the Measurement Set.
        unsigned int row = r + start_row;
        uvw(0) = uu[r]; uvw(1) = vv[r]; uvw(2) = ww[r];
        col_uvw.put(row, uvw);
        col_antenna1.put(row, p->a1[r]);
        col_antenna2.put(row, p->a2[r]);
        col_weight.put(row, weight);
        col_sigma.put(row, sigma);
        col_exposure.put(row, exposure_sec);
        col_interval.put(row, interval_sec);
        col_time.put(row, time_stamp);
        col_timeCentroid.put(row, time_stamp);
    }

    // Update time range if required.
    if (time_stamp < p->start_time)
    {
        p->start_time = time_stamp - interval_sec/2.0;
    }
    if (time_stamp > p->end_time)
    {
        p->end_time = time_stamp + interval_sec/2.0;
    }
    p->time_inc_sec = interval_sec;
    p->data_written = 1;
}

void oskar_ms_write_coords_d(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int num_baselines,
        const double* uu, const double* vv, const double* ww,
        double exposure_sec, double interval_sec, double time_stamp)
{
    oskar_ms_write_coords(p, start_row, num_baselines, uu, vv, ww,
            exposure_sec, interval_sec, time_stamp);
}

void oskar_ms_write_coords_f(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int num_baselines,
        const float* uu, const float* vv, const float* ww,
        double exposure_sec, double interval_sec, double time_stamp)
{
    oskar_ms_write_coords(p, start_row, num_baselines, uu, vv, ww,
            exposure_sec, interval_sec, time_stamp);
}

template <typename T>
void oskar_ms_write_vis(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int start_channel,
        unsigned int num_channels, unsigned int num_baselines, const T* vis)
{
    // Allocate storage for the block of visibility data.
    unsigned int num_pols = p->num_pols;
    IPosition shape(3, num_pols, num_channels, num_baselines);
    Array<Complex> vis_data(shape);

    // Copy visibility data into the array,
    // swapping baseline and channel dimensions.
    float* out = (float*) vis_data.data();
    for (unsigned int c = 0; c < num_channels; ++c)
    {
        for (unsigned int b = 0; b < num_baselines; ++b)
        {
            for (unsigned int p = 0; p < num_pols; ++p)
            {
                unsigned int i = (num_pols * (c * num_baselines + b) + p) << 1;
                unsigned int j = (num_pols * (b * num_channels + c) + p) << 1;
                out[j]     = vis[i];
                out[j + 1] = vis[i + 1];
            }
        }
    }

    // Add new rows if required.
    oskar_ms_ensure_num_rows(p, start_row + num_baselines);

    // Create the slicers for the column.
    IPosition start1(1, start_row);
    IPosition length1(1, num_baselines);
    Slicer row_range(start1, length1);
    IPosition start2(2, 0, start_channel);
    IPosition length2(2, num_pols, num_channels);
    Slicer array_section(start2, length2);

    // Write visibilities to DATA column.
#ifdef OSKAR_MS_NEW
    ArrayColumn<Complex>& col_data = p->msmc.data;
#else
    ArrayColumn<Complex>& col_data = p->msmc->data();
#endif
    col_data.putColumnRange(row_range, array_section, vis_data);
    p->data_written = 1;
}

void oskar_ms_write_vis_d(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int start_channel,
        unsigned int num_channels, unsigned int num_baselines,
        const double* vis)
{
    oskar_ms_write_vis(p, start_row, start_channel,
            num_channels, num_baselines, vis);
}

void oskar_ms_write_vis_f(oskar_MeasurementSet* p,
        unsigned int start_row, unsigned int start_channel,
        unsigned int num_channels, unsigned int num_baselines,
        const float* vis)
{
    oskar_ms_write_vis(p, start_row, start_channel,
            num_channels, num_baselines, vis);
}
